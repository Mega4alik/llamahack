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
    from huggingface_hub import HfApi, HfFolder, Repository, create_repo
    # Set repo name (your-username/my-model-name)
    repo_name = "AnuarSh/landing1"

    # Create the repo (or skip if already created)    
    #create_repo(repo_name, private=False)
    
    api = HfApi()
    api.upload_folder(
        folder_path="./model_temp/checkpoint-24000",
        repo_id=repo_name,
        repo_type="model",
    )


if __name__=="__main__":
    hf_push_to_hub()
    