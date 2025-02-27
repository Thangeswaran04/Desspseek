import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

class TextGenerationPipeline {
  static model_id = "onnx-community/Phi-3.5-mini-instruct-onnx-web";

  static async getInstance(progress_callback = null) {
    this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
      progress_callback,
    });

    this.model ??= AutoModelForCausalLM.from_pretrained(this.model_id, {
      dtype: "q4f16",
      device: "webgpu",
      use_external_data_format: true,
      progress_callback,
    });

    return Promise.all([this.tokenizer, this.model]);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();

// Persistent Memory Storage
let memory = JSON.parse(localStorage.getItem("chat_memory")) || [];
async function generate(messages) {
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();

  // Append past conversations
  messages = [...memory, ...messages];

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });

  let startTime;
  let numTokens = 0;
  let tps;
  const token_callback_function = () => {
    startTime ??= performance.now();
    if (numTokens++ > 0) {
      tps = (numTokens / (performance.now() - startTime)) * 1000;
    }
  };

  const streamer = new TextStreamer(token_callback_function);
  const output = await model.generate(inputs.input_ids, {
    stopping_criteria,
    streamer,
  });

  const response = tokenizer.decode(output[0], { skip_special_tokens: true });

  // Store response in memory
  memory.push({ role: "assistant", content: response });
  localStorage.setItem("chat_memory", JSON.stringify(memory));

  return response;
}
