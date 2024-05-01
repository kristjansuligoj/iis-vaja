from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.report import Report
from evidently.tests import *
from definitions import ROOT_DIR

import pandas as pd
import sys
import os


def main():
    reference_data = pd.read_csv(ROOT_DIR + "/data/reference_data.csv")
    current_data = pd.read_csv(ROOT_DIR + "/data/current_data.csv")

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Create directory for reports if it doesn't exist
    reports_dir = os.path.join(ROOT_DIR, "reports", "data_tests")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    report.save(os.path.join(reports_dir, "data_drift.json"))

    tests = TestSuite(tests=[
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=reference_data, current_data=current_data)
    test_results = tests.as_dict()

    # Check if any test failed
    if test_results['summary']['failed_tests'] > 0:
        print("Some tests failed:")
        print(test_results['summary'])

        sys.exit(1)
    else:
        print("All tests passed!")

    # Create directory for reports if it doesn't exist
    reports_dir = os.path.join(ROOT_DIR, "reports", "sites")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    tests.save_html(os.path.join(reports_dir, "stability_tests.html"))


if __name__ == "__main__":
    main()
