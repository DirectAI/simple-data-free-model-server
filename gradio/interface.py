import json
import numpy as np
import gradio as gr  # type: ignore[import-untyped]
from functools import partial
from typing import Union, Tuple
from PIL import Image

from modeling import deploy_classifier, get_classifier_results

# to add later maybe
# my_include {border: 2px solid green !important}
# my_exclude {border: 2px solid red !important}

css = """
#my_accordion {background-color: #0B0F19}
.feedback textarea {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo:
    class_states = gr.State([])
    current_class_idx = gr.State(0)
    current_accordion_open = gr.State(False)

    def upload_file(bytes: bytes) -> Union[dict, list]:
        json_data = json.loads(bytes.decode("utf-8"))
        print(json_data)
        return json_data

    with gr.Row():
        with gr.Column():
            with gr.Accordion("classifier definition", open=True):
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
                count_val = len(class_states_val)

                for i in range(count_val):

                    def update_class_label(
                        class_idx: int,
                        class_name: str,
                        class_states_val: list = class_states_val,
                    ) -> list:
                        class_states_val[class_idx]["name"] = class_name
                        if len(class_states_val[class_idx]["examples_to_include"]) == 0:
                            class_states_val[class_idx]["examples_to_include"] = [
                                class_name
                            ]
                        return class_states_val

                    def update_include_example(
                        class_idx: int,
                        include_idx: int,
                        label_name: str,
                        class_states_val: list = class_states_val,
                    ) -> None:
                        class_states_val[class_idx]["examples_to_include"][
                            include_idx
                        ] = label_name

                    def update_exclude_example(
                        class_idx: int,
                        exclude_idx: int,
                        label_name: str,
                        class_states_val: list = class_states_val,
                    ) -> None:
                        class_states_val[class_idx]["examples_to_exclude"][
                            exclude_idx
                        ] = label_name

                    def change_example_count(
                        class_idx: int,
                        type_of_example: str,
                        change_key: str,
                        class_states_val: list = class_states_val,
                    ) -> Tuple[list, int, bool]:
                        # We expect `type_of_example` to be "to_include" or "to_exclude"
                        # We expect `change_key` to be "increment" or "decrement"
                        if change_key == "increment":
                            if type_of_example == "examples_to_include":
                                class_states_val[class_idx][type_of_example].append(
                                    f"to include in class {class_idx}"
                                )
                            elif type_of_example == "examples_to_exclude":
                                class_states_val[class_idx][type_of_example].append(
                                    f"to exclude from class {class_idx}"
                                )

                        elif change_key == "decrement":
                            class_states_val[class_idx][type_of_example] = (
                                class_states_val[class_idx][type_of_example][:-1]
                            )
                        return class_states_val, class_idx, True

                    class_label = class_states_val[i]["name"]
                    to_include_list = class_states_val[i]["examples_to_include"]
                    to_exclude_list = class_states_val[i]["examples_to_exclude"]

                    with gr.Group(class_label, elem_id=f"tab_{i}"):
                        box = gr.Textbox(
                            key=i,
                            label="class label",
                            value=class_label,
                            autofocus=True,
                        )
                        box.submit(
                            fn=partial(update_class_label, i),
                            inputs=[box],
                            outputs=[class_states],
                        )
                        with gr.Accordion(
                            "advanced configuration",
                            elem_id="my_accordion",
                            open=i == class_idx and is_accordion_open,
                        ):
                            with gr.Accordion(
                                "examples to include", elem_id="my_include", open=True
                            ):
                                with gr.Row():
                                    add_inc_example_btn = gr.Button("Add Example")
                                    add_inc_example_btn.click(
                                        partial(
                                            change_example_count,
                                            i,
                                            "examples_to_include",
                                            "increment",
                                        ),
                                        None,
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
                                        None,
                                        [
                                            class_states,
                                            current_class_idx,
                                            current_accordion_open,
                                        ],
                                    )
                                for j, inc_example in enumerate(to_include_list):
                                    curr_inc = gr.Textbox(
                                        key=f"inc_{i}_{j}", label="", value=inc_example
                                    )
                                    curr_inc.submit(
                                        fn=partial(update_include_example, i, j),
                                        inputs=[curr_inc],
                                    )

                            with gr.Accordion(
                                "examples to exclude", elem_id="my_exclude", open=True
                            ):
                                with gr.Row():
                                    add_exc_example_btn = gr.Button("Add Example")
                                    add_exc_example_btn.click(
                                        partial(
                                            change_example_count,
                                            i,
                                            "examples_to_exclude",
                                            "increment",
                                        ),
                                        None,
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
                                        None,
                                        [
                                            class_states,
                                            current_class_idx,
                                            current_accordion_open,
                                        ],
                                    )
                                for j, exc_example in enumerate(to_exclude_list):
                                    curr_exc = gr.Textbox(
                                        key=f"exc_{i}_{j}", label="", value=exc_example
                                    )
                                    curr_exc.submit(
                                        fn=partial(update_exclude_example, i, j),
                                        inputs=[curr_exc],
                                    )

                model_builder_button = gr.Button("Export JSON")
                model_json_textbox = gr.JSON(
                    label="Model JSON", value=class_states.value
                )

                def update_model_json(class_s: list) -> list:
                    return class_s

                model_builder_button.click(
                    update_model_json, class_states, model_json_textbox
                )

        with gr.Column():

            def placeholder_fn(
                to_display: Union[Image.Image, np.ndarray], set_of_classes: list
            ) -> dict:
                deployed_id = deploy_classifier(set_of_classes)
                classify_results = get_classifier_results(to_display, deployed_id)
                return classify_results

            img_to_display = gr.Image()
            to_output = gr.Label()
            img_to_display.input(
                placeholder_fn, [img_to_display, class_states], to_output
            )


demo.launch(server_name="0.0.0.0")
