from launch_vllm import launch, get_gpu_count


def main(mounts: list[str] = [], vllm_args: list[str] = []):

    # Launch vLLM on all gpus
    gpus = range(get_gpu_count())
    with launch(
        gpus=gpus, 
        model_name="ByteDance-Seed/UI-TARS-1.5-7B",
        mounts=mounts,
        vllm_args=vllm_args
    ) as proc:
        # Wait until ready

        # Generate rollout batch

        # Stop vllm
        pass

    # Launch training on this batch

    # Loop

    pass



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument()
    args = parser.parse_args()
    main(**vars(args))