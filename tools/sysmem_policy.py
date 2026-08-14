"""CUDA - Sysmem Fallback Policy をコマンドラインから読み書きしようとする試み.

    python tools/sysmem_policy.py            # 現在の設定を読むだけ
    python tools/sysmem_policy.py --apply 1  # 値を書き込んで保存する

なぜこれが要るか。VRAM を使い切ったとき、Windows のドライバは既定で
システムRAM へこぼす。落ちないので気付けないまま10倍近く遅くなる。
NVIDIA コントロールパネルで「Prefer No Sysmem Fallback」にすれば
こぼす代わりに OOM で落ちてくれるが、GUI 操作は手順書とスクショが要る。
自動化できるなら読者の再現性が上がる。

なぜ普通のレジストリ操作では書けないか。NVIDIA の3D設定は
`C:\\ProgramData\\NVIDIA Corporation\\Drs\\nvdrsdb0.bin` という独自形式の
データベースに入っている。レジストリではないので reg add では触れない。
公式の入口は NVAPI の DRS (Driver Settings) API だけである。

設定IDは推測しない。ドライバに登録されている設定を全部列挙して、
名前に Sysmem を含むものを探す。当てずっぽうで別の設定を書き換えるのが
いちばん危ないので、必ず名前で引き当ててから触る。
"""

from __future__ import annotations

import argparse
import ctypes
import sys

NVAPI_OK = 0
NVAPI_UNICODE_STRING_MAX = 2048
NVAPI_BINARY_DATA_MAX = 4096
NVAPI_SETTING_MAX_VALUES = 100

# nvapi_QueryInterface に渡す関数ID。公開 SDK の nvapi_interface テーブルの値。
# 間違っていれば QueryInterface が NULL を返すだけなので、誤った操作にはならない。
FUNC_IDS = {
    "NvAPI_Initialize": 0x0150E828,
    "NvAPI_Unload": 0xD22BDD7E,
    "NvAPI_GetErrorMessage": 0x6C2D048C,
    "NvAPI_DRS_CreateSession": 0x0694D52E,
    "NvAPI_DRS_DestroySession": 0xDAD9CFF8,
    "NvAPI_DRS_LoadSettings": 0x375DBD6B,
    "NvAPI_DRS_SaveSettings": 0xFCBC7E14,
    "NvAPI_DRS_GetBaseProfile": 0xDA8466A0,
    "NvAPI_DRS_GetSetting": 0x73BF8338,
    "NvAPI_DRS_SetSetting": 0x577DD202,
    "NvAPI_DRS_EnumAvailableSettingIds": 0xF020614A,
    "NvAPI_DRS_GetSettingNameFromId": 0xD61CBE6E,
    "NvAPI_DRS_EnumAvailableSettingValues": 0x2EC39F90,
}

NVDRS_DWORD_TYPE = 0
_LOCATION_NAMES = {
    0: "現在のプロファイル",
    1: "グローバル",
    2: "ベースプロファイル",
    3: "既定値",
}

UnicodeString = ctypes.c_uint16 * NVAPI_UNICODE_STRING_MAX


class BinarySetting(ctypes.Structure):
    _fields_ = [
        ("valueLength", ctypes.c_uint32),
        ("valueData", ctypes.c_uint8 * NVAPI_BINARY_DATA_MAX),
    ]


class _ValueUnion(ctypes.Union):
    _fields_ = [
        ("u32Value", ctypes.c_uint32),
        ("binaryValue", BinarySetting),
        ("wszValue", UnicodeString),
    ]


class NvdrsSetting(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("settingName", UnicodeString),
        ("settingId", ctypes.c_uint32),
        ("settingType", ctypes.c_uint32),
        ("settingLocation", ctypes.c_uint32),
        ("isCurrentPredefined", ctypes.c_uint32),
        ("isPredefinedValid", ctypes.c_uint32),
        ("predefined", _ValueUnion),
        ("current", _ValueUnion),
    ]


class NvdrsSettingValues(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("numSettingValues", ctypes.c_uint32),
        ("settingType", ctypes.c_uint32),
        ("defaultValue", _ValueUnion),
        ("settingValues", _ValueUnion * NVAPI_SETTING_MAX_VALUES),
    ]


def make_version(struct_type, version: int) -> int:
    """NVAPI の構造体は先頭に「サイズ | 版番号<<16」を入れる約束になっている."""
    return ctypes.sizeof(struct_type) | (version << 16)


def to_text(buffer) -> str:
    chars = []
    for code in buffer:
        if code == 0:
            break
        chars.append(chr(code))
    return "".join(chars)


def to_unicode_string(text: str) -> UnicodeString:
    buffer = UnicodeString()
    for i, char in enumerate(text[: NVAPI_UNICODE_STRING_MAX - 1]):
        buffer[i] = ord(char)
    return buffer


class NvApi:
    def __init__(self) -> None:
        self.dll = ctypes.WinDLL("nvapi64.dll")
        self.dll.nvapi_QueryInterface.restype = ctypes.c_void_p
        self.dll.nvapi_QueryInterface.argtypes = [ctypes.c_uint32]
        self._cache: dict[str, object] = {}
        self.missing: list[str] = []

    def fn(self, name: str, *argtypes):
        if name in self._cache:
            return self._cache[name]
        address = self.dll.nvapi_QueryInterface(FUNC_IDS[name])
        if not address:
            self.missing.append(name)
            raise NvApiError(f"{name} を QueryInterface で取れませんでした")
        prototype = ctypes.WINFUNCTYPE(ctypes.c_int32, *argtypes)
        func = prototype(address)
        self._cache[name] = func
        return func

    def error_text(self, status: int) -> str:
        try:
            buffer = ctypes.create_string_buffer(64)
            self.fn("NvAPI_GetErrorMessage", ctypes.c_int32, ctypes.c_char_p)(
                status, buffer
            )
            return buffer.value.decode("ascii", "replace")
        except Exception:
            return f"status {status}"

    def check(self, status: int, what: str) -> None:
        if status != NVAPI_OK:
            raise NvApiError(f"{what} が失敗しました: {self.error_text(status)} ({status})")


class NvApiError(RuntimeError):
    pass


def open_session(api: NvApi) -> ctypes.c_void_p:
    api.check(api.fn("NvAPI_Initialize")(), "NvAPI_Initialize")
    session = ctypes.c_void_p()
    api.check(
        api.fn("NvAPI_DRS_CreateSession", ctypes.POINTER(ctypes.c_void_p))(
            ctypes.byref(session)
        ),
        "NvAPI_DRS_CreateSession",
    )
    api.check(
        api.fn("NvAPI_DRS_LoadSettings", ctypes.c_void_p)(session),
        "NvAPI_DRS_LoadSettings",
    )
    return session


def enumerate_settings(api: NvApi) -> list[tuple[int, str]]:
    """ドライバに登録されている設定のIDと名前を全部返す."""
    count = ctypes.c_uint32(0)
    enum = api.fn(
        "NvAPI_DRS_EnumAvailableSettingIds",
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    )
    status = enum(None, ctypes.byref(count))
    if count.value == 0:
        api.check(status, "NvAPI_DRS_EnumAvailableSettingIds (件数の問い合わせ)")
    ids = (ctypes.c_uint32 * count.value)()
    api.check(enum(ids, ctypes.byref(count)), "NvAPI_DRS_EnumAvailableSettingIds")

    name_of = api.fn(
        "NvAPI_DRS_GetSettingNameFromId", ctypes.c_uint32, ctypes.POINTER(UnicodeString)
    )
    found = []
    for setting_id in ids[: count.value]:
        buffer = UnicodeString()
        if name_of(setting_id, ctypes.byref(buffer)) != NVAPI_OK:
            continue
        found.append((setting_id, to_text(buffer)))
    return found


def find_candidates(api: NvApi, keywords: tuple[str, ...]) -> list[tuple[int, str]]:
    """列挙した設定から、名前がキーワードに当たるものを絞る."""
    settings = enumerate_settings(api)
    print(f"  ドライバに登録されている設定 : {len(settings)} 件")
    return [
        (i, name)
        for i, name in settings
        if any(keyword in name.lower() for keyword in keywords)
    ]


def read_setting(api: NvApi, session, profile, setting_id: int) -> NvdrsSetting | None:
    setting = NvdrsSetting()
    setting.version = make_version(NvdrsSetting, 1)
    status = api.fn(
        "NvAPI_DRS_GetSetting",
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(NvdrsSetting),
    )(session, profile, setting_id, ctypes.byref(setting))
    if status != NVAPI_OK:
        print(f"    現在値の取得: {api.error_text(status)} "
              "(プロファイルに未設定なら既定値のままという意味)")
        return None
    return setting


def read_available_values(api: NvApi, setting_id: int) -> list[int]:
    values = NvdrsSettingValues()
    values.version = make_version(NvdrsSettingValues, 1)
    status = api.fn(
        "NvAPI_DRS_EnumAvailableSettingValues",
        ctypes.c_uint32,
        ctypes.POINTER(NvdrsSettingValues),
    )(setting_id, ctypes.byref(values))
    if status != NVAPI_OK:
        print(f"    選べる値の列挙: {api.error_text(status)}")
        return []
    if values.settingType != NVDRS_DWORD_TYPE:
        print(f"    選べる値の列挙: 数値型ではありません (type {values.settingType})")
        return []
    print(f"    既定値      : {values.defaultValue.u32Value}")
    return [values.settingValues[i].u32Value for i in range(values.numSettingValues)]


def apply_setting(api: NvApi, session, profile, setting_id: int, name: str, value: int) -> bool:
    setting = NvdrsSetting()
    setting.version = make_version(NvdrsSetting, 1)
    setting.settingId = setting_id
    setting.settingType = NVDRS_DWORD_TYPE
    setting.settingName = to_unicode_string(name)
    setting.current.u32Value = value
    status = api.fn(
        "NvAPI_DRS_SetSetting",
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(NvdrsSetting),
    )(session, profile, ctypes.byref(setting))
    if status != NVAPI_OK:
        print(f"  SetSetting が失敗: {api.error_text(status)} ({status})")
        return False
    status = api.fn("NvAPI_DRS_SaveSettings", ctypes.c_void_p)(session)
    if status != NVAPI_OK:
        print(f"  SaveSettings が失敗: {api.error_text(status)} ({status})")
        print("  nvdrsdb0.bin への書き込みには管理者権限が必要です。")
        return False
    print(f"  {name} を {value} にして保存しました。")
    return True


GUI_INSTRUCTIONS = """\
コマンドラインから設定できなかった場合の手順 (GUI)

  1. デスクトップを右クリック → NVIDIA コントロールパネル
  2. 左の「3D設定の管理」を開く
  3. 「グローバル設定」タブの一覧から
     「CUDA - システムメモリ フォールバック ポリシー」
     (英語表示なら CUDA - Sysmem Fallback Policy) を探す
  4. 値を「システムメモリ フォールバックを優先しない」
     (Prefer No Sysmem Fallback) に変える
  5. 右下の「適用」を押す
  6. Python のプロセスを起動し直す (既に動いているプロセスには効かない)

設定できたかどうかは python check_env.py で確認できる。
「VRAM 超過時の挙動」が OutOfMemoryError になれば効いている。
共有GPUメモリの増分が出るなら効いていない。

設定できなくても学習は進められる。runtime.configure(0.85) と MemoryGuard で
PyTorch 側から蓋をしてあるので、こぼれる前に自分で止まる。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply",
        type=int,
        default=None,
        metavar="VALUE",
        help="この値を書き込んで保存する (省略時は読むだけ)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="DRS に登録されている設定を全部並べる (自分のドライバを確かめる用)",
    )
    args = ap.parse_args()

    print("=" * 74)
    print("  CUDA - Sysmem Fallback Policy を NVAPI から扱う")
    print("=" * 74)

    if sys.platform != "win32":
        print("Windows 専用です。")
        return 1

    try:
        api = NvApi()
    except OSError as exc:
        print(f"nvapi64.dll を読めませんでした: {exc}")
        print()
        print(GUI_INSTRUCTIONS)
        return 1

    session = None
    try:
        session = open_session(api)
        print("  NVAPI の初期化と DRS セッション : 成功")

        profile = ctypes.c_void_p()
        api.check(
            api.fn(
                "NvAPI_DRS_GetBaseProfile",
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(session, ctypes.byref(profile)),
            "NvAPI_DRS_GetBaseProfile",
        )
        print("  ベースプロファイルの取得        : 成功")

        if args.list:
            for setting_id, name in enumerate_settings(api):
                print(f"  0x{setting_id:08X}  {name}")
            return 0

        candidates = find_candidates(api, ("sysmem", "fallback", "system memory"))
        if not candidates:
            names = enumerate_settings(api)
            cuda_related = [n for _i, n in names if "cuda" in n.lower()]
            print("  名前に sysmem / fallback を含む設定はありませんでした。")
            print(f"  名前に cuda を含む設定も {len(cuda_related)} 件です。")
            print()
            print("  列挙された設定はすべて 3D 描画側 (アンチエイリアス・DLSS・G-SYNC など)")
            print("  でした。CUDA - Sysmem Fallback Policy は NVIDIA コントロールパネルに")
            print("  出てくるのに、NvAPI_DRS_EnumAvailableSettingIds には登録されていません。")
            print("  つまり公開 NVAPI からは自動設定できません。")
            print()
            print("  設定IDを推測して SetSetting を呼べば書き込み自体はできますが、")
            print("  当たっている保証が無いまま別のドライバ設定を書き換える危険があるので")
            print("  やりません。--list で自分のドライバの一覧を確認できます。")
            print()
            print(GUI_INSTRUCTIONS)
            return 1

        print(f"  該当する設定                    : {len(candidates)} 件")
        for setting_id, name in candidates:
            print()
            print(f"  0x{setting_id:08X}  {name}")
            values = read_available_values(api, setting_id)
            if values:
                print(f"    選べる値    : {values}")
            setting = read_setting(api, session, profile, setting_id)
            if setting is not None:
                print(f"    現在値      : {setting.current.u32Value}")
                print(f"    設定元      : "
                      f"{_LOCATION_NAMES.get(setting.settingLocation, setting.settingLocation)}")
                print(f"    既定のままか: {'はい' if setting.isCurrentPredefined else 'いいえ'}")

        if args.apply is None:
            print()
            print("読み取りのみで終了しました。書き込むには --apply <値> を付けます。")
            print("どの値が「こぼさない」側かは、書き込んだあとに")
            print("python check_env.py で実測して確かめてください。")
            return 0

        setting_id, name = candidates[0]
        print()
        print(f"  {name} に {args.apply} を書き込みます")
        if not apply_setting(api, session, profile, setting_id, name, args.apply):
            print()
            print(GUI_INSTRUCTIONS)
            return 1
        print()
        print("Python を起動し直してから python check_env.py で効果を確認してください。")
        return 0

    except NvApiError as exc:
        print(f"  失敗: {exc}")
        if api.missing:
            print(f"  取れなかった関数: {', '.join(api.missing)}")
        print()
        print(GUI_INSTRUCTIONS)
        return 1
    finally:
        if session is not None:
            try:
                api.fn("NvAPI_DRS_DestroySession", ctypes.c_void_p)(session)
                api.fn("NvAPI_Unload")()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
