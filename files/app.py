import streamlit as st
import torch
import numpy as np
from Synthesizer import SynthesizerTrn
from your_script_name import preprocess_text, TextMapper, utils  # Adjust the import according to your script organization

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configurations and model
ckpt_dir = r"files\G_100000.pth"
vocab_file = r"files\vocab.txt"
config_file = r"files\config.json"
assert os.path.isfile(config_file), f"{config_file} doesn't exist"
hps = utils.get_hparams_from_file(config_file)
text_mapper = TextMapper(vocab_file)
net_g = SynthesizerTrn(
    len(text_mapper.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
net_g.to(device)
net_g.eval()

# Load model weights
g_pth = r"files\G_100000.pth"
utils.load_checkpoint(g_pth, net_g, None)

# Streamlit interface
st.title("Text to Speech Converter")
input_text = st.text_area("Enter text to convert", "Type your text here...")
lang_option = st.selectbox("Select Language", options=['English', 'ron'], index=0)  # Adjust according to your needs

if st.button("Convert to Speech"):
    processed_text = preprocess_line(input_text, text_mapper, hps, lang=lang_option)
    stn_tst = text_mapper.get_text(processed_text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio_output = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_comp=0.8, length_scale=1.0)[0][0,0].cpu().float().numpy()
    st.audio(audio_output, format='audio/wav', start_time=0)

st.write("Generated audio will appear above after clicking 'Convert to Speech'")
