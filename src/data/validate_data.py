from definitions import ROOT_DIR
import great_expectations
import sys


def main():
    try:
        data_context = great_expectations.DataContext(context_root_dir=(ROOT_DIR + "/gx"))
    except great_expectations.exceptions.DataContextError as e:
        print(f"Error loading data context: {e}")
        sys.exit(1)

    validation = data_context.run_checkpoint(
        checkpoint_name="merged_data_checkpoint",
        batch_request=None,
        run_name=None,
    )

    if not validation["success"]:
        print("Validation failed!")
        sys.exit(1)

    print("Validation succeeded!")


if __name__ == "__main__":
    main()
