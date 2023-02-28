import gradio as gr
import numpy as np
from audioldm import text_to_audio, build_model

# from share_btn import community_icon_html, loading_icon_html, share_js

model_id = "haoheliu/AudioLDM-S-Full"

audioldm = build_model()
# audioldm=None

# def predict(input, history=[]):
#     # tokenize the new input sentence
#     new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

#     # generate a response
#     history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

#     # convert the tokens to text, and then split the responses into lines
#     response = tokenizer.decode(history[0]).split("<|endoftext|>")
#     response = [(response[i], response[i+1]) for i in range(0, len(response)-1, 2)]  # convert to tuples of list
#     return response, history

def text2audio(text, duration, guidance_scale, random_seed, n_candidates):
    # print(text, length, guidance_scale)
    waveform = text_to_audio(
        latent_diffusion=audioldm,
        text=text,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]
    waveform = [
        gr.make_waveform((16000, wave[0]), bg_image="bg.png") for wave in waveform
    ]
    # waveform = [(16000, np.random.randn(16000)), (16000, np.random.randn(16000))]
    if len(waveform) == 1:
        waveform = waveform[0]
    return waveform


# iface = gr.Interface(fn=text2audio, inputs=[
#         gr.Textbox(value="A man is speaking in a huge room", max_lines=1),
#         gr.Slider(2.5, 10, value=5, step=2.5),
#         gr.Slider(0, 5, value=2.5, step=0.5),
#         gr.Number(value=42)
#     ], outputs=[gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")],
#                 allow_flagging="never"
#                      )
# iface.launch(share=True)


css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #000000;
            background: #000000;
        }
        input[type='range'] {
            accent-color: #000000;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #generated_id{
            min-height: 700px
        }
        #setting_id{
          margin-bottom: 12px;
          text-align: center;
          font-weight: 900;
        }
"""
iface = gr.Blocks(css=css)

with iface:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  AudioLDM: Text-to-Audio Generation with Latent Diffusion Models
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                <a href="https://arxiv.org/abs/2301.12503">[Paper]</a>  <a href="https://audioldm.github.io/">[Project page]</a>
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            ############# Input
            textbox = gr.Textbox(
                value="A hammer is hitting a wooden surface",
                max_lines=1,
                label="Input your text here. Please ensure it is descriptive and of moderate length.",
                elem_id="prompt-in",
            )

            with gr.Accordion("Click to modify detailed configurations", open=False):
                seed = gr.Number(
                    value=42,
                    label="Change this value (any integer number) will lead to a different generation result.",
                )
                duration = gr.Slider(
                    2.5, 10, value=5, step=2.5, label="Duration (seconds)"
                )
                guidance_scale = gr.Slider(
                    0,
                    5,
                    value=2.5,
                    step=0.5,
                    label="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
                )
                n_candidates = gr.Slider(
                    1,
                    5,
                    value=3,
                    step=1,
                    label="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
                )
            ############# Output
            # outputs=gr.Audio(label="Output", type="numpy")
            outputs = gr.Video(label="Output", elem_id="output-video")

            # with gr.Group(elem_id="container-advanced-btns"):
            #   # advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            #   with gr.Group(elem_id="share-btn-container"):
            #     community_icon = gr.HTML(community_icon_html, visible=False)
            #     loading_icon = gr.HTML(loading_icon_html, visible=False)
            #     share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
            # outputs=[gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")]
            btn = gr.Button("Submit").style(full_width=True)

        # with gr.Group(elem_id="share-btn-container", visible=False):
        #     community_icon = gr.HTML(community_icon_html)
        #     loading_icon = gr.HTML(loading_icon_html)
        #     share_button = gr.Button("Share to community", elem_id="share-btn")

        btn.click(
            text2audio,
            inputs=[textbox, duration, guidance_scale, seed, n_candidates],
            outputs=[outputs],
        )

        # share_button.click(None, [], [], _js=share_js)
        gr.HTML(
            """
        <div class="footer" style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <p>Follow the latest update of AudioLDM on our<a href="https://github.com/haoheliu/AudioLDM" style="text-decoration: underline;" target="_blank"> Github repo</a>
                    </p>
                    <br>
                    <p>Model by <a href="https://twitter.com/LiuHaohe" style="text-decoration: underline;" target="_blank">Haohe Liu</a></p>
                    <br>
        </div>
        """
        )
        gr.Examples(
            [
                ["A hammer is hitting a wooden surface", 5, 2.5, 45, 3],
                [
                    "Peaceful and calming ambient music with singing bowl and other instruments.",
                    5,
                    2.5,
                    45,
                    3,
                ],
                ["A man is speaking in a small room.", 5, 2.5, 45, 3],
                ["A female is speaking followed by footstep sound", 5, 2.5, 45, 3],
                [
                    "Wooden table tapping sound followed by water pouring sound.",
                    5,
                    2.5,
                    45,
                    3,
                ],
            ],
            fn=text2audio,
            inputs=[textbox, duration, guidance_scale, seed, n_candidates],
            outputs=[outputs],
            cache_examples=True,
        )
        with gr.Accordion("Additional information", open=False):
            gr.HTML(
                """
                <div class="acknowledgments">
                    <p> We build the model with data from <a href="http://research.google.com/audioset/">AudioSet</a>, <a href="https://freesound.org/">Freesound</a> and <a href="https://sound-effects.bbcrewind.co.uk/">BBC Sound Effect library</a>. We share this demo based on the <a href="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/375954/Research.pdf">UK copyright exception</a> of data for academic research. </p>
                            </div>
                        """
            )
# <p>This demo is strictly for research demo purpose only. For commercial use please <a href="haoheliu@gmail.com">contact us</a>.</p>

iface.queue(concurrency_count=3)
# iface.launch(debug=True)
iface.launch(debug=True, share=True)
