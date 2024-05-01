from definitions import ROOT_DIR
import great_expectations
import sys


def main():
    data_context = great_expectations.DataContext(context_root_dir=(ROOT_DIR + "/gx"))

    validation = data_context.run_checkpoint(checkpoint_name="merged_data_checkpoint")

    if not validation["success"]:
        print("Validation failed!")
        sys.exit(1)

    print("Validation succeeded!")


if __name__ == "__main__":
    main()
