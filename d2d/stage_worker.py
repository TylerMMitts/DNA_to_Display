# Runs one stage's analysis in its own process, for base.run's memory guard
# (resources.run_guarded). Writes the stage's report.json; exit code 0 on success.
#
#     python -m d2d.stage_worker <dataset> <stage>
import sys
import time
import traceback

from .project import Dataset, write_json


def main():
    name, stage_name = sys.argv[1], sys.argv[2]
    from .stages.base import get_stage
    ds = Dataset(name)
    stage = get_stage(stage_name)
    t0 = time.time()
    try:
        report = stage.analyze(ds)
    except SystemExit as e:          # a stage's own "cannot run" message, not a crash
        print(f'D2D_STAGE_STOP: {e}', flush=True)
        sys.exit(2)
    except BaseException:            # noqa: BLE001 - the parent reads the traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
    write_json(ds.stage_dir(stage_name) / 'report.json', report)
    print(f'analysis took {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
