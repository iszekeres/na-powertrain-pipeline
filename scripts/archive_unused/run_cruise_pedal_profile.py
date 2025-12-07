from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))

import tools.cruise_pedal_profile as profile


def run_for_file(path_str: str) -> None:
    print(f"\n[RUN] cruise_pedal_profile on {path_str}")
    profile.CSV_PATH = Path(path_str)
    profile.main()


if __name__ == "__main__":
    for filename in ["newlogs/17-gpt1-1.csv", "newlogs/17-gpt1-2.csv"]:
        run_for_file(filename)
