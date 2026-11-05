"""
Run several CADS tasks on one CT in ONE predict call.

CADS's own script (cads/scripts/predict_images.py) accepts a single task or
'all' per call, so for 7 tasks the CT is preprocessed (reorientation,
resampling to 1.5 mm) and restored 7 times. cads.utils.inference.predict
itself takes a list of tasks and preprocesses once - this script calls it
directly. It is started with the Python of the CADS environment
(CADS_COMMAND[0]) and only imports CADS, nothing from this project.

    <CADS python> cads_predict_multi.py -in CT.nii.gz -out <dir> -tasks 551,552,559 -license research [--cpu]

Output as with the CADS script: <dir>/<case>/<case>_part_<task>.nii.gz
"""
import argparse
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)


def main():
    p = argparse.ArgumentParser(description="CADS: several tasks in one run")
    p.add_argument("-in", dest="input", required=True, help="CT .nii.gz")
    p.add_argument("-out", dest="output", required=True)
    p.add_argument("-tasks", required=True, help="comma-separated task ids, e.g. 551,552,559")
    p.add_argument("-license", default="research", choices=["reference", "research", "open"])
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    tasks = sorted({int(t) for t in a.tasks.split(",") if t.strip()})

    from cads.utils.libs import setup_nnunet_env
    setup_nnunet_env(a.license)
    from cads.utils.inference import predict
    from cads.utils.libs import check_or_download_model_weights, get_model_weights_dir

    # weights, incl. the pre-segmentations 557/558 need (as in predict_images.py)
    for task in tasks:
        check_or_download_model_weights(task, a.license)
    if any(t in tasks for t in (557, 558)) and 553 not in tasks:
        check_or_download_model_weights(553, a.license)
        if 558 in tasks and 552 not in tasks:
            check_or_download_model_weights(552, a.license)

    predict([a.input], a.output, get_model_weights_dir(a.license), tasks, folds="all",
            use_cpu=a.cpu, preprocess_cads=True, postprocess_cads=True,
            save_all_combined_seg=False, snapshot=False, save_separate_targets=False, mode="auto")


if __name__ == "__main__":
    main()