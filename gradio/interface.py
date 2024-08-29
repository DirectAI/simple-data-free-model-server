import json
import os
import io
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
    classify_deploy_and_infer,
    detect_deploy_and_infer,
    update_class_detection_threshold,
    update_models_state,
    update_nms_threshold,
)

from illustrating import display_bounding_boxes

from modeling import DualModelInterface, ModelType

from typing import Union

css = """
#configuration_accordion {background-color: #0B0F19}
"""

with gr.Blocks(css=css) as demo:
    models_state = gr.State(DualModelInterface())
    # TLDR: Gradio State isn't properly managing complex state variables so we track model changes by pairing with a boolean
    # See message in Gradio discord: https://discord.com/channels/879548962464493619/1278443437175345272/1278443437175345272
    # when gradio releases a fix we can remove all mentions of `change_this_bool_to_force_reload` and maintain existing functionality
    change_this_bool_to_force_reload = gr.State(False)
    current_class_idx = gr.State(0)
    current_class_accordion_open = gr.State(False)

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                label="Type of Model", choices=["Detector", "Classifier"]
            )
            model_dropdown.change(
                update_models_state,
                inputs=[model_dropdown, models_state, change_this_bool_to_force_reload],
                outputs=[models_state, change_this_bool_to_force_reload],
            )

            @gr.render(
                inputs=[
                    models_state,
                    current_class_idx,
                    current_class_accordion_open,
                    change_this_bool_to_force_reload,
                ],
            )
            def render_model_update(
                models_state_val: DualModelInterface,
                class_idx: int,
                is_class_accordion_open: bool,
                proxy_bool: bool,
            ) -> None:
                if models_state_val.current_model_type is not None:
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
                                change_this_bool_to_force_reload,
                            ],
                            outputs=[models_state, change_this_bool_to_force_reload],
                        )
                        with gr.Row():
                            add_btn = gr.Button("Add Class")
                            sub_btn = gr.Button("Remove Class")
                            add_btn.click(
                                lambda m_state, proxy_bool: m_state.add_class(
                                    proxy_bool
                                ),
                                inputs=[models_state, change_this_bool_to_force_reload],
                                outputs=[
                                    models_state,
                                    change_this_bool_to_force_reload,
                                ],
                            )
                            sub_btn.click(
                                lambda m_state, proxy_bool: m_state.remove_class(
                                    proxy_bool
                                ),
                                inputs=[models_state, change_this_bool_to_force_reload],
                                outputs=[
                                    models_state,
                                    change_this_bool_to_force_reload,
                                ],
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
                                    change_this_bool_to_force_reload,
                                ],
                                outputs=[
                                    models_state,
                                    change_this_bool_to_force_reload,
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
                                                change_this_bool_to_force_reload,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                                change_this_bool_to_force_reload,
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
                                                change_this_bool_to_force_reload,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                                change_this_bool_to_force_reload,
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
                                                change_this_bool_to_force_reload,
                                            ],
                                            outputs=[
                                                models_state,
                                                change_this_bool_to_force_reload,
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
                                                change_this_bool_to_force_reload,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                                change_this_bool_to_force_reload,
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
                                                change_this_bool_to_force_reload,
                                            ],
                                            [
                                                models_state,
                                                current_class_idx,
                                                current_class_accordion_open,
                                                change_this_bool_to_force_reload,
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
                                                change_this_bool_to_force_reload,
                                            ],
                                            outputs=[
                                                models_state,
                                                change_this_bool_to_force_reload,
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
                                            change_this_bool_to_force_reload,
                                        ],
                                        outputs=[
                                            models_state,
                                            current_class_idx,
                                            current_class_accordion_open,
                                            change_this_bool_to_force_reload,
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
                                change_this_bool_to_force_reload,
                            ],
                            outputs=[models_state, change_this_bool_to_force_reload],
                        )

                    with gr.Accordion("Export JSON", open=True):
                        model_json_textbox = gr.JSON(
                            value=models_state_val.display_dict(),
                        )

        with gr.Column():

            @gr.render(
                inputs=[models_state, change_this_bool_to_force_reload],
                triggers=[
                    models_state.change,
                    change_this_bool_to_force_reload.change,
                    # TODO: add demo launch as a trigger
                ],
            )
            def render_image(
                models_state_val: DualModelInterface,
                proxy_bool: bool,
            ) -> None:
                def del_image() -> None:
                    if os.path.exists("temp.jpg"):
                        os.remove("temp.jpg")

                def save_image(
                    gradio_img: Union[Image.Image, np.ndarray, str]
                ) -> Image.Image:
                    print(type(gradio_img))
                    if isinstance(gradio_img, np.ndarray):
                        img_byte_arr = io.BytesIO()
                        pil_image = Image.fromarray(gradio_img)
                    elif isinstance(gradio_img, str):
                        pil_image = Image.open(gradio_img)
                    print(type(pil_image))
                    pil_image.save("temp.jpg", format="JPEG")
                    if models_state_val.current_model_type == ModelType.DETECTOR:
                        inference_results = detect_deploy_and_infer(
                            to_display=pil_image, models_state_val=models_state_val
                        )
                        print(inference_results)
                        illustrated_image = display_bounding_boxes(
                            pil_image=pil_image,
                            dets=inference_results[0],
                            threshold=0.1,
                        )
                        return illustrated_image
                    elif models_state_val.current_model_type == ModelType.CLASSIFIER:
                        inference_results = classify_deploy_and_infer(
                            img_to_display.value, models_state_val
                        )
                        output_label = gr.Label(inference_results)
                        return pil_image
                    else:
                        gr.Info(
                            "No valid model config. Please select Detector or Classifier from the Dropdown.",
                            duration=5,
                        )
                        return pil_image

                # if os.path.exists("temp.jpg"):
                #     pil_image = Image.open("temp.jpg")
                #     if models_state_val.current_model_type == ModelType.DETECTOR:
                #         inference_results = detect_deploy_and_infer(
                #             to_display = pil_image,
                #             models_state_val = models_state_val
                #         )
                #         print(inference_results)
                #         illustrated_image = display_bounding_boxes(
                #             gradio_img = pil_image,
                #             dets= inference_results[0],
                #             threshold= 0.1,
                #         )
                #         img_to_display = gr.Image(value=illustrated_image)
                #     # render the image at the illustrated fp
                #     else:
                #         img_to_display = gr.Image(value=pil_image)
                # else:
                img_to_display = gr.Image()

                img_to_display.clear(del_image)
                img_to_display.input(
                    save_image, inputs=img_to_display, outputs=img_to_display
                )

            # @gr.render(
            #     inputs=[models_state, change_this_bool_to_force_reload],
            #     triggers=[
            #         models_state.change,
            #         change_this_bool_to_force_reload.change,
            #     ],
            # )
            # def render_results(
            #     models_state_val: DualModelInterface,
            #     proxy_bool: bool,
            # ) -> None:
            #     img_to_display = gr.Image(key="img_to_display")
            #     # img_to_display.input(

            #     # )
            #     if img_to_display.value is not None:
            #         if models_state_val.current_model_type == ModelType.DETECTOR:
            #             inference_results = detect_deploy_and_infer(
            #                 img_to_display.value, models_state_val
            #             )
            #             print(inference_results)
            # elif models_state_val.current_model_type == ModelType.CLASSIFIER:
            #     inference_results = classify_deploy_and_infer(
            #         img_to_display.value, models_state_val
            #     )
            #     _ = gr.Label(inference_results)
            # else:
            #     gr.Info(
            #         "No valid model config. Please select Detector or Classifier from the Dropdown.",
            #         duration=5,
            #     )


demo.launch(server_name="0.0.0.0")
