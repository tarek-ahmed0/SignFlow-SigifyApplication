# Hugging Face LLM
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
translation_pipeline = pipeline("translation",
                                model = "facebook/nllb-200-distilled-600M",
                                tokenizer = "facebook/nllb-200-distilled-600M")

# Saving The LLM Model & Tokenizer After Downloading :
model = translation_pipeline.model.save_pretrained("llm_translation_model")
tokenizer = translation_pipeline.tokenizer.save_pretrained("llm_translation_model")

# Loading The LLM Full Pipeline From Local :
def intiate_pipeline(path = "llm_translation_model") :
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return pipeline("translation", model = model, tokenizer = tokenizer)

# Initiating The Pipeline :
translation_pipeline = intiate_pipeline()
translation_pipeline

# Build The Translation Function ( To Import In Our Main APP )
def coll2formal_translation(processed_text, pipeline = translation_pipeline) :
    translated_text = pipeline(processed_text, src_lang = "arz_Arab", tgt_lang = "arb_Arab")
    return translated_text[0]["translation_text"]