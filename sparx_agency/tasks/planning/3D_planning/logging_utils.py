# interactive_rrtstar/logging_utils.py
def pinfo(msg: str) -> None:
    print(f"[INFO] {msg}")


def pok(msg: str) -> None:
    print(f"[OK]   {msg}")


def pwarn(msg: str) -> None:
    print(f"[WARN] {msg}")


def perr(msg: str) -> None:
    print(f"[ERR]  {msg}")
