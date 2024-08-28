import json
import gradio as gr  # type: ignore[import-untyped]
from functools import partial
from typing import Tuple
from PIL import Image
import numpy as np


from utils import (
    upload_file,
    update_class_label,
    update_include_example,
    update_exclude_example,
    change_example_count,
    deploy_and_infer,
    update_class_detection_threshold,
    update_models_state,
    update_nms_threshold,
)

from modeling import DualModelInterface

from typing import Union

css = """
#configuration_accordion {background-color: #0B0F19}
"""

with gr.Blocks(css=css) as demo:
    models_state = gr.State(DualModelInterface())
    models_state_proxy = gr.State(False)
    current_class_idx = gr.State(0)
    current_accordion_open = gr.State(False)

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                label="Type of Model", choices=["Detector", "Classifier"]
            )
            model_dropdown.change(
                update_models_state,
                inputs=[model_dropdown, models_state, models_state_proxy],
                outputs=[models_state, models_state_proxy],
            )

            @gr.render(
                inputs=[
                    models_state,
                    current_class_idx,
                    current_accordion_open,
                    models_state_proxy,
                ],
            )
            def render_count(
                models_state_val: DualModelInterface,
                class_idx: int,
                is_accordion_open: bool,
                models_state_proxy_val: bool,
            ) -> None:
                if models_state_val.current_model_type != "":
                    with gr.Group("Model Definition"):
                        json_upload_button = gr.UploadButton(
                            "Import JSON",
                            file_count="single",
                            file_types=["json"],
                            type="binary",
                        )
                        json_upload_button.upload(
                            upload_file,
                            inputs=[
                                json_upload_button,
                                models_state,
                                models_state_proxy,
                            ],
                            outputs=[models_state, models_state_proxy],
                        )
                        with gr.Row():
                            add_btn = gr.Button("Add Class")
                            sub_btn = gr.Button("Remove Class")
                            add_btn.click(
                                lambda x, y: (x.add_class(), not y),
                                inputs=[models_state, models_state_proxy],
                                outputs=[models_state, models_state_proxy],
                            )
                            sub_btn.click(
                                lambda x, y: (x.remove_class(), not y),
                                inputs=[models_state, models_state_proxy],
                                outputs=[models_state, models_state_proxy],
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
                                inputs=[box, models_state, models_state_proxy],
                                outputs=[models_state, models_state_proxy],
                            )
                            with gr.Accordion(
                                "advanced configuration",
                                elem_id="configuration_accordion",
                                open=i == class_idx and is_accordion_open,
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
                                            [models_state, models_state_proxy],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_accordion_open,
                                                models_state_proxy,
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
                                            [models_state, models_state_proxy],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_accordion_open,
                                                models_state_proxy,
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
                                                models_state_proxy,
                                            ],
                                            outputs=[
                                                models_state,
                                                models_state_proxy,
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
                                            [models_state, models_state_proxy],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_accordion_open,
                                                models_state_proxy,
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
                                            [models_state, models_state_proxy],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_accordion_open,
                                                models_state_proxy,
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
                                                models_state_proxy,
                                            ],
                                            outputs=[models_state, models_state_proxy],
                                        )
                                if models_state_val.current_model_type == "Detector":
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
                                            models_state_proxy,
                                        ],
                                        outputs=[models_state, models_state_proxy],
                                    )

                    if models_state_val.current_model_type == "Detector":
                        nms_threshold = gr.Slider(
                            minimum=0,
                            maximum=1.0,
                            value=0.4,
                            label="NMS Threshold",
                            interactive=True,
                        )
                        nms_threshold.release(
                            update_nms_threshold,
                            inputs=[models_state, nms_threshold, models_state_proxy],
                            outputs=[models_state, models_state_proxy],
                        )

                    with gr.Group("Model Export"):
                        model_builder_button = gr.Button("Export JSON")
                        model_json_textbox = gr.JSON(
                            label="Model JSON", value=models_state_val.dict()
                        )

                        model_builder_button.click(
                            lambda x: x.dict(), models_state, model_json_textbox
                        )

        with gr.Column():
            img_to_display = gr.Image()

            @gr.render(
                inputs=[models_state, img_to_display, models_state_proxy],
                triggers=[
                    models_state.change,
                    img_to_display.input,
                    models_state_proxy.change,
                ],
            )
            def render_results(
                models_state_val: DualModelInterface,
                img_to_display: Union[Image.Image, np.ndarray],
                models_state_proxy_val: bool,
            ) -> None:
                if img_to_display is not None:
                    inference_results, error_text = deploy_and_infer(
                        img_to_display, models_state_val
                    )
                    _ = gr.Label(inference_results)
                    _ = gr.Markdown(error_text)


demo.launch(server_name="0.0.0.0")
