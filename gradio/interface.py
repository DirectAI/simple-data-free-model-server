import json

import gradio as gr  # type: ignore[import-untyped]
from functools import partial

from PIL import Image
import numpy as np

from utils import (
    upload_file,
    update_class_label,
    update_include_example,
    update_exclude_example,
    change_example_count,
    deploy_and_infer,
)

from typing import Union

css = """
#configuration_accordion {background-color: #0B0F19}
"""

with gr.Blocks(css=css) as demo:
    class_states = gr.State([])
    current_class_idx = gr.State(0)
    current_accordion_open = gr.State(False)

    with gr.Row():
        with gr.Column():
            with gr.Group("classifier definition"):
                json_upload_button = gr.UploadButton(
                    "Import JSON",
                    file_count="single",
                    file_types=["json"],
                    type="binary",
                )
                json_upload_button.upload(upload_file, json_upload_button, class_states)
                with gr.Row():
                    add_btn = gr.Button("Add Class")
                    sub_btn = gr.Button("Remove Class")

                add_btn.click(
                    lambda x: x
                    + [
                        {
                            "name": f"class_{len(x)}",
                            "examples_to_include": [],
                            "examples_to_exclude": [],
                        }
                    ],
                    class_states,
                    class_states,
                )
                sub_btn.click(lambda x: x[:-1], class_states, class_states)

            @gr.render(inputs=[class_states, current_class_idx, current_accordion_open])
            def render_count(
                class_states_val: list, class_idx: int, is_accordion_open: bool
            ) -> None:
                print("rendering count!", class_states_val)
                count_val = len(class_states_val)

                for i in range(count_val):
                    class_label = class_states_val[i]["name"]
                    to_include_list = class_states_val[i]["examples_to_include"]
                    to_exclude_list = class_states_val[i]["examples_to_exclude"]
                    print(class_label, to_include_list, to_exclude_list)

                    with gr.Group(class_label, elem_id=f"tab_{i}"):
                        box = gr.Textbox(
                            label="class label",
                            value=class_label,
                            autofocus=True,
                        )
                        box.submit(
                            fn=partial(update_class_label, i),
                            inputs=[box, class_states],
                            outputs=[class_states],
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
                                        [class_states],
                                        [
                                            class_states,
                                            current_class_idx,
                                            current_accordion_open,
                                        ],
                                    )
                                    rem_inc_example_btn = gr.Button("Remove Example")
                                    rem_inc_example_btn.click(
                                        partial(
                                            change_example_count,
                                            i,
                                            "examples_to_include",
                                            "decrement",
                                        ),
                                        [class_states],
                                        [
                                            class_states,
                                            current_class_idx,
                                            current_accordion_open,
                                        ],
                                    )
                                for j, inc_example in enumerate(to_include_list):
                                    curr_inc = gr.Textbox(label="", value=inc_example)
                                    curr_inc.submit(
                                        fn=partial(update_include_example, i, j),
                                        inputs=[curr_inc, class_states],
                                        outputs=[class_states],
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
                                        [class_states],
                                        [
                                            class_states,
                                            current_class_idx,
                                            current_accordion_open,
                                        ],
                                    )
                                    rem_exc_example_btn = gr.Button("Remove Example")
                                    rem_exc_example_btn.click(
                                        partial(
                                            change_example_count,
                                            i,
                                            "examples_to_exclude",
                                            "decrement",
                                        ),
                                        [class_states],
                                        [
                                            class_states,
                                            current_class_idx,
                                            current_accordion_open,
                                        ],
                                    )
                                for j, exc_example in enumerate(to_exclude_list):
                                    curr_exc = gr.Textbox(label="", value=exc_example)
                                    curr_exc.submit(
                                        fn=partial(update_exclude_example, i, j),
                                        inputs=[curr_exc, class_states],
                                        outputs=[class_states],
                                    )

                model_builder_button = gr.Button("Export JSON")
                model_json_textbox = gr.JSON(
                    label="Model JSON", value=class_states.value
                )

                model_builder_button.click(
                    lambda x: x, class_states, model_json_textbox
                )

        with gr.Column():
            img_to_display = gr.Image()

            @gr.render(
                inputs=[class_states, img_to_display],
                triggers=[class_states.change, img_to_display.input],
            )
            def render_results(
                class_states_val: list, img_to_display: Union[Image.Image, np.ndarray]
            ) -> None:
                print("render_results!")
                print(class_states_val, type(img_to_display))
                if img_to_display is not None:
                    print("image isn't none!")
                    inference_results, error_text = deploy_and_infer(
                        img_to_display, class_states_val
                    )
                    _ = gr.Label(inference_results)
                    _ = gr.Markdown(error_text)


demo.launch(server_name="0.0.0.0")
