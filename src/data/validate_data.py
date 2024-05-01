import great_expectations
import sys


def main():
    data_context = great_expectations.get_context()

    valid_data = data_context.run_checkpoint(
        checkpoint_name="merged_data_checkpoint",
        batch_request=None,
        run_name=None,
    )

    if not valid_data["success"]:
        print("Data is invalid!")
        sys.exit(1)

    print("Data is valid!")


if __name__ == "__main__":
    main()
