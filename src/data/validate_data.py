from definitions import ROOT_DIR
import great_expectations


def main():
    data_context = great_expectations.DataContext(context_root_dir=ROOT_DIR + "/gx")

    validation = data_context.run_checkpoint(checkpoint_name="merged_data_checkpoint", batch_request=None, run_name=None)

    if validation["success"]:
        print("Validation succeeded!")

    print("Validation failed!")
    return


if __name__ == "__main__":
    main()
