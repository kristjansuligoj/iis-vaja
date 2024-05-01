from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.report import Report
from evidently.tests import *

import pandas as pd
import sys


def main():
    reference_data = pd.read_csv("../../data/reference_data.csv")
    current_data = pd.read_csv("../../data/current_data.csv")

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    report.save('../../reports/data_tests/data_drift.json')

    tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
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

    tests.save_html("reports/sites/stability_tests.html")


if __name__ == "__main__":
    main()
