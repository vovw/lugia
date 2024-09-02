from model import *
if __name__ == "__main__":
    model_path = "mlx_model"
    prompt = "write quicksort in python:"
    max_tokens = 100
    temp = 0.0
    tokens_per_eval = 10
    seed = 0

    mx.random.seed(seed)
    print("[INFO] Loading model from disk.")
    model, tokenizer = load_model(model_path)

    print("[INFO] Starting generation...")
    tic = time.time()
    print(prompt, end="", flush=True)
    prompt_encoded = mx.array(tokenizer.encode(prompt))
    tokens = []
    for token, ntoks in zip(generate(prompt_encoded, model, temp), range(max_tokens)):
        tokens.append(token)
        if ntoks == 0:
            mx.eval(tokens)
            toc = time.time()
            prompt_tps = prompt_encoded.size / (toc - tic)
            tic = time.time()
        if (len(tokens) % tokens_per_eval) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
    print("------")
    generation_tps = ntoks / (time.time() - tic)
    print(f"Tokens per second: prompt {prompt_tps:.3f}, generation {generation_tps:.3f}")
