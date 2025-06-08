def hf_inference():
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        provider="hf-inference",
        api_key="<hf_api_key>",
    )

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
    )

    print(completion.choices[0].message)


#=========================================


def hf_push_to_hub():
    from huggingface_hub import HfApi, HfFolder, Repository

    # Set repo name (your-username/my-model-name)
    repo_name = "AnuarSh/landing1"

    # Create the repo (or skip if already created)
    from huggingface_hub import create_repo
    create_repo(repo_name, private=False)

    # Clone the repo locally
    from huggingface_hub import snapshot_download
    from huggingface_hub import Repository

    repo = Repository(local_dir="./model_temp/checkpoint-19000", clone_from=repo_name)

    # Push your model directory
    repo.push_to_hub()



if __name__=="__main__":
    #hf_push_to_hub()
    pass