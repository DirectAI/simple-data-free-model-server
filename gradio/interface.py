import json
import uuid
import gradio as gr  # type: ignore[import-untyped]
from functools import partial
from PIL import Image
import numpy as np
import requests


from utils import (
    upload_file,
    upload_json_data,
    update_class_label,
    update_include_example,
    update_exclude_example,
    change_example_count,
    update_class_detection_threshold,
    update_models_state,
    update_nms_threshold,
    dual_model_infer,
    dual_model_deploy,
    deploy_to_production,
)

from modeling import DualModelInterface, ModelType

from typing import Union, Tuple

css = """
#configuration_accordion {background-color: #0B0F19}
"""

with gr.Blocks(css=css) as demo:
    frontend_ephemeral_deployed_id = gr.State(None)
    models_state = gr.State(DualModelInterface())
    current_class_idx = gr.State(0)
    current_class_accordion_open = gr.State(False)

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                label="Type of Model", choices=["Detector", "Classifier"]
            )
            model_dropdown.change(
                update_models_state,
                inputs=[model_dropdown, models_state],
                outputs=[models_state],
            )

            @gr.render(
                inputs=[models_state, current_class_idx, current_class_accordion_open],
            )
            def render_model_update(
                models_state_val: DualModelInterface,
                class_idx: int,
                is_class_accordion_open: bool,
            ) -> None:
                if models_state_val.current_model_type is not None:
                    with gr.Accordion("External Import", open=False):
                        json_upload_button = gr.UploadButton(
                            "Import JSON",
                            file_count="single",
                            file_types=["json"],
                            type="binary",
                        )
                        json_upload_button.upload(
                            upload_file,
                            inputs=[json_upload_button, models_state],
                            outputs=[models_state],
                        )
                        _ = gr.Markdown(
                            "<div style='text-align: center;'><b>OR</b></div>"
                        )
                        model_uuid = gr.Textbox(label="Import Model via UUID")
                        model_uuid_button = gr.Button("Submit UUID")
                        model_uuid_button.click(
                            upload_json_data,
                            inputs=[model_uuid, models_state],
                            outputs=[models_state],
                        )

                    with gr.Row():
                        add_btn = gr.Button("Add Class")
                        sub_btn = gr.Button("Remove Class")
                        add_btn.click(
                            lambda m_state: m_state.add_class(),
                            inputs=[models_state],
                            outputs=[models_state],
                        )
                        sub_btn.click(
                            lambda m_state: m_state.remove_class(),
                            inputs=[models_state],
                            outputs=[models_state],
                        )
                    count_val = len(models_state_val)

                    for i in range(count_val):
                        class_label = models_state_val.get_class_label(i)
                        to_include_list = models_state_val.get_class_extoinc(i)
                        to_exclude_list = models_state_val.get_class_extoexc(i)

                        with gr.Group(class_label, elem_id=f"tab_{i}"):
                            box = gr.Textbox(
                                label="class label",
                                value=class_label,
                                autofocus=True,
                            )
                            box.submit(
                                fn=partial(update_class_label, i),
                                inputs=[
                                    box,
                                    models_state,
                                ],
                                outputs=[
                                    models_state,
                                ],
                            )
                            with gr.Accordion(
                                "advanced configuration",
                                elem_id="configuration_accordion",
                                open=i == class_idx and is_class_accordion_open,
                            ):
                                with gr.Accordion("examples to include", open=True):
                                    with gr.Row():
                                        add_inc_example_btn = gr.Button("Add Example")
                                        add_inc_example_btn.click(
                                            partial(
                                                change_example_count,
                                                i,
                                                "examples_to_include",
                                                "increment",
                                            ),
                                            [
                                                models_state,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                            ],
                                        )
                                        rem_inc_example_btn = gr.Button(
                                            "Remove Example"
                                        )
                                        rem_inc_example_btn.click(
                                            partial(
                                                change_example_count,
                                                i,
                                                "examples_to_include",
                                                "decrement",
                                            ),
                                            [
                                                models_state,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                            ],
                                        )
                                    for j, inc_example in enumerate(to_include_list):
                                        curr_inc = gr.Textbox(
                                            label="", value=inc_example
                                        )
                                        curr_inc.submit(
                                            fn=partial(update_include_example, i, j),
                                            inputs=[
                                                curr_inc,
                                                models_state,
                                            ],
                                            outputs=[
                                                models_state,
                                            ],
                                        )

                                with gr.Accordion("examples to exclude", open=True):
                                    with gr.Row():
                                        add_exc_example_btn = gr.Button("Add Example")
                                        add_exc_example_btn.click(
                                            partial(
                                                change_example_count,
                                                i,
                                                "examples_to_exclude",
                                                "increment",
                                            ),
                                            [
                                                models_state,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                            ],
                                        )
                                        rem_exc_example_btn = gr.Button(
                                            "Remove Example"
                                        )
                                        rem_exc_example_btn.click(
                                            partial(
                                                change_example_count,
                                                i,
                                                "examples_to_exclude",
                                                "decrement",
                                            ),
                                            [
                                                models_state,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                            ],
                                        )
                                    for j, exc_example in enumerate(to_exclude_list):
                                        curr_exc = gr.Textbox(
                                            label="", value=exc_example
                                        )
                                        curr_exc.submit(
                                            fn=partial(update_exclude_example, i, j),
                                            inputs=[
                                                curr_exc,
                                                models_state,
                                            ],
                                            outputs=[
                                                models_state,
                                            ],
                                        )
                                if (
                                    models_state_val.current_model_type
                                    == ModelType.DETECTOR
                                ):
                                    class_threshold = gr.Slider(
                                        minimum=0,
                                        maximum=1.0,
                                        value=models_state_val.detector_state.detector_configs[
                                            i
                                        ].detection_threshold,
                                        label="Class Threshold",
                                        step=0.01,
                                        interactive=True,
                                    )
                                    class_threshold.release(
                                        partial(update_class_detection_threshold, i),
                                        inputs=[
                                            models_state,
                                            class_threshold,
                                        ],
                                        outputs=[
                                            models_state,
                                            current_class_idx,
                                            current_class_accordion_open,
                                        ],
                                    )

                    if models_state_val.current_model_type == ModelType.DETECTOR:
                        nms_threshold = gr.Slider(
                            minimum=0,
                            maximum=1.0,
                            value=models_state_val.detector_state.nms_threshold,
                            label="NMS Threshold",
                            interactive=True,
                        )
                        nms_threshold.release(
                            update_nms_threshold,
                            inputs=[
                                models_state,
                                nms_threshold,
                            ],
                            outputs=[models_state],
                        )

                    with gr.Accordion("Export JSON", open=True):
                        model_json_textbox = gr.JSON(
                            value=models_state_val.display_dict(),
                        )
                        deploy_config_button = gr.Button("Deploy JSON to Production")
                        deploy_config_button.click(
                            deploy_to_production,
                            inputs=models_state,
                            outputs=[models_state, model_json_textbox],
                        )

        with gr.Column():
            raw_img_to_hide = gr.Image(visible=False)
            with gr.Accordion("Upload an Image via URL", open=False) as url_accordion:
                optional_img_url = gr.Textbox(label="", placeholder="your://url/here")
                img_url_submit_button = gr.Button("Submit")

            upload_option_markdown = gr.Markdown(
                "<div style='text-align: center;'><b>OR</b></div>"
            )
            img_to_display = gr.Image()
            classification_label = gr.Label(visible=False)
            img_url_submit_button.click(
                lambda img_url: (
                    Image.open(requests.get(img_url, stream=True).raw),
                    gr.Accordion("Upload an Image via URL", open=False),
                    gr.Textbox(label="", placeholder="your://url/here", value=""),
                ),
                [optional_img_url],
                [raw_img_to_hide, url_accordion, optional_img_url],
            ).then(
                dual_model_infer,
                inputs=[raw_img_to_hide, models_state, frontend_ephemeral_deployed_id],
                outputs=[
                    img_to_display,
                    classification_label,
                    frontend_ephemeral_deployed_id,
                ],
            )
            classification_label = gr.Label(visible=False)

            # if we change the image input, save it to the ~hidden~ image component for later use
            img_to_display.input(
                lambda x: gr.Image(x, visible=False),
                inputs=[img_to_display],
                outputs=[raw_img_to_hide],
            ).then(
                dual_model_infer,
                inputs=[raw_img_to_hide, models_state, frontend_ephemeral_deployed_id],
                outputs=[
                    img_to_display,
                    classification_label,
                    frontend_ephemeral_deployed_id,
                ],
            )

            # if we update the model, run inference again (on the raw image)
            # note that once gradio fixes its image issue, we should be monitoring the model state, not the proxy boolean
            models_state.change(
                dual_model_infer,
                inputs=[raw_img_to_hide, models_state, frontend_ephemeral_deployed_id],
                outputs=[
                    img_to_display,
                    classification_label,
                    frontend_ephemeral_deployed_id,
                ],
            )

demo.launch(server_name="0.0.0.0")
