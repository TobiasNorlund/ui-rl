from litellm import completion
import json
from tqdm import tqdm


PROMPT_1 = """The following is an instruction to a GUI agent, to be performed on a computer using the GUI:

--------
{system_prompt}
--------

The agent has already performed the following actions:
{prev_actions}

Here is the current screenshot:"""

PROMPT_2 = """This is the next output: 

{next_output}

I want you to write this output, in the same format and with the same action, but with a paraphrazed Thought, i.e. it should motivate the action in a correct way. Just output this, Thought and Action, on separate lines, and nothing else."""


def generate_variant(completion_idx: int, messages):
    prev_actions = ""
    idx = 1
    for i in range(completion_idx):
        if messages[i]["role"] == "assistant":
            prev_actions += f"{idx}:\n{messages[i]["content"]}\n\n"
            idx += 1

    last_screenshot_message = messages[completion_idx-1]["content"][0]
    assert last_screenshot_message["type"] == "image_url"

    next_output = messages[completion_idx]["content"]
    assert type(next_output) == str

    while True:
        resp = completion(
            model="gemini/gemini-2.5-flash",
            messages=[{
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": PROMPT_1.format(
                            system_prompt=messages[0]["content"],
                            prev_actions=prev_actions
                        )
                    },
                    last_screenshot_message,
                    {
                        "type": "text",
                        "text": PROMPT_2.format(next_output=next_output)
                    }
                ]
            }]
        )

        content = resp["choices"][0]["message"]["content"]
        if content.count("Action:") == 1:
            break
        else:
            breakpoint()

    return content


def main(rollout: str, n: int, output :str):
    with open(rollout) as f:
        rollout = json.load(f)

    # Generate N variants of each completion text
    variants = []
    for completion in tqdm(rollout["completions"], desc="Completion", position=1):
        variants.append([])
        for _ in tqdm(range(n), desc="n", position=2, leave=False):
            variants[-1].append(generate_variant(completion["generated_message"], rollout["messages"]))

    print(f"Generated {n} variants:\n")
    for completion, variations in zip(rollout["completions"], variants):
        print(f"ORIGINAL:")
        orig = rollout["messages"][completion["generated_message"]]["content"]
        print(orig)
        print()
        for variant in variations:
            print(variant)
            print()
        print("---")
        print()

    for completion, variations in zip(rollout["completions"], variants):
        # Replace original content
        rollout["messages"][completion["generated_message"]]["content"] = [
            {"type": "text", "text": [orig] + variations}
        ]
        # Remove token ids from completion
        del completion["prompt_token_ids"]
        del completion["generated_token_ids"]

    with open(output, "w") as f:
        json.dump(rollout, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout")
    parser.add_argument("--n", type=int, default=1, help="num variants per completion")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(**vars(args))