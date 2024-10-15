def load_base_model(base_model):
    base_path = base_model  # input: base model

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token = "<unk>"
    tokenizer.padding_side = "right"
    

    quantization_config_loading = GPTQConfig(
        bits=4,
        disable_exllama=True,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        quantization_config=quantization_config_loading,
        device_map="auto",
        torch_dtype=torch.float16,
    )

   #set pad and eos token 
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id #surely need to set it in the actual model as well? not just the config
    model.resize_token_embeddings(len(tokenizer))


    return tokenizer,model

#class app_config:

#    def __init__.(self)