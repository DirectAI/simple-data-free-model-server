import ray
import numpy as np
from collections import defaultdict

from modeling.batch_processing import (
    build_ray_dataset_from_directory,
    run_image_classifier_against_ray_dataset,
    write_predictions_to_csv,
)


def load_labels_dict_from_csv(
    file_path: str, strip_dir_name: bool = False, strip_extension: bool = False
) -> dict[str, dict[str, str]]:
    labels_dict = {}
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                index_to_attribute_name = line.strip().split(",")
            else:
                entries = line.strip().split(",")
                data_path = entries[0]
                if strip_dir_name:
                    data_path = data_path.split("/")[-1]
                if strip_extension:
                    data_path = data_path.rsplit(".", 1)[0]
                attributes = {
                    index_to_attribute_name[j]: entries[j]
                    for j in range(1, len(entries))
                }
                labels_dict[data_path] = attributes
    return labels_dict


def add_attribute_to_dataset_as_label(
    dataset: ray.data.Dataset,
    labels_dict: dict[str, dict[str, str]],
    attribute_name: str,
) -> ray.data.Dataset:
    def add_attribute_as_label(
        row: dict[str, np.ndarray | str]
    ) -> dict[str, np.ndarray | str]:
        img_path = row["path"]
        assert isinstance(img_path, str)
        row["label"] = labels_dict.get(img_path, {}).get(attribute_name, "")
        return row

    return dataset.map(add_attribute_as_label)


def compute_average_value_conditional_on_attributes(
    predictions: ray.data.Dataset,
    value_name: str,
    labels_dict: dict[str, dict[str, str]],
    attribute_names: list[str],
) -> dict[str | tuple[str, ...], float]:
    value_sums: dict[tuple[str, ...], float] = defaultdict(float)
    value_counts: dict[tuple[str, ...], int] = defaultdict(int)

    for prediction in predictions.iter_rows():
        attributes_dict = labels_dict.get(prediction["path"], {})
        # attribute_value = labels_dict.get(prediction["path"], {}).get(attribute_name, "")
        attribute_values = tuple(
            attributes_dict.get(attribute_name, "")
            for attribute_name in attribute_names
        )
        value_sums[attribute_values] += prediction[value_name]
        value_counts[attribute_values] += 1

    average_values: dict[str | tuple[str, ...], float] = {
        key: value_sums[key] / value_counts[key] for key in value_sums
    }

    # sort keys alphabetically
    average_values = dict(sorted(average_values.items()))

    # add overall score to end
    balanced_average = sum(average_values.values()) / len(average_values)
    average_values["Overall_Balanced"] = balanced_average
    overall_average = sum(value_sums.values()) / sum(value_counts.values())
    average_values["Overall_Count"] = overall_average

    return average_values


def filter_dataset_by_attribute(
    dataset: ray.data.Dataset,
    labels_dict: dict[str, dict[str, str]],
    attribute_name: str,
    attribute_values: list[str],
) -> ray.data.Dataset:
    attribute_values_set = set(attribute_values)

    def filter_by_attribute(row: dict[str, np.ndarray | str]) -> bool:
        img_path = row["path"]
        assert isinstance(img_path, str)
        return (
            labels_dict.get(img_path, {}).get(attribute_name, "")
            in attribute_values_set
        )

    return dataset.filter(filter_by_attribute)


if __name__ == "__main__":
    # Load labels from CSV file
    labels_dict = load_labels_dict_from_csv(
        "/directai_fastapi/.cache/fairface/fairface_label_train.csv",
        strip_dir_name=True,
        strip_extension=True,
    )

    # Load dataset
    dataset = build_ray_dataset_from_directory(
        "/directai_fastapi/.cache/fairface/train", with_subdirs_as_labels=False
    )

    # # limit dataset for testing purposes
    # dataset = dataset.limit(10000)

    # Add attribute to dataset as label
    dataset = add_attribute_to_dataset_as_label(dataset, labels_dict, "gender")

    # # create basic gender classifier configs
    # gender_labels = ["Male", "Female", "Black Male", "Black Female", "Asian Male", "Asian Female"]
    # # gender_include_labels_dict = {
    # #     "Male": ["Male",],
    # #     "Female": ["Female",],
    # # }
    # gender_include_labels_dict = {
    #     label: [label,] for label in gender_labels
    # }

    # # setup classifier run
    # predictions = run_image_classifier_against_ray_dataset(
    #     dataset,
    #     batch_size=64,
    #     labels=gender_labels,
    #     inc_sub_labels_dict=gender_include_labels_dict,
    # )

    # # # compute accuracy
    # # accuracy = predictions.mean("is_correct")
    # # print(f"Accuracy: {accuracy}")
    # # # accuracy on training split for naive config: 0.943

    # # # write predictions to CSV
    # # write_predictions_to_csv(predictions, "/directai_fastapi/.cache/fairface/fairface_train_pred.csv")

    # # # compute accuracy conditional on race
    # # conditional_gender_accuracy = compute_average_value_conditional_on_attribute(
    # #     predictions,
    # #     "is_correct",
    # #     labels_dict,
    # #     "race",
    # # )
    # # print(conditional_gender_accuracy)
    # # # {'East Asian': 0.9446569545047612, 'Indian': 0.9440701355629515, 'Latino_Hispanic': 0.9515972170270068, 'Southeast Asian': 0.9345067160722557, 'Middle Eastern': 0.9626736111111112, 'Black': 0.8995340472492438, 'White': 0.9587946995825014}

    # # compute average value of raw scores conditional on race
    # conditional_raw_scores = compute_average_value_conditional_on_attribute(
    #     predictions,
    #     "raw_scores",
    #     labels_dict,
    #     "race",
    # )
    # print({key: {gender_labels[i]: value[i] for i in range(len(gender_labels))} for key, value in conditional_raw_scores.items()})

    gender_labels = ["Male", "Female"]
    augmented_gender_include_labels_dict = {
        "Male": [
            # "Male",
            # "black man",
            "man",
            # "east asian man",
            # "southeast asian man",
            # "indian man",
            # "asian man",
            # "hispanic man",
            # "arabic man",
            # "white man",
            # "old man",
            # "old black man",
            # "black boy",
            # "man with long hair",
            # "man with short hair",
            # "baby boy",
            # "teenage boy",
            # "boy in his 20s",
            # "boy in his 30s",
            # "middle age man",
            # "old man",
            "boy aged 0-2",
            "boy aged 3-9",
            "boy aged 10-19",
            "man aged 20-29",
            "man aged 30-39",
            "man aged 40-49",
            "man aged 50-59",
            "man aged 60-69",
            "man aged more than 70",
        ],
        "Female": [
            # "Female",
            # "black woman",
            "woman",
            # "east asian woman",
            # "southeast asian woman",
            # "indian woman",
            # "asian woman",
            # "hispanic woman",
            # "arabic woman",
            # "white woman",
            # "old woman",
            # "old black woman",
            # "black girl",
            # "woman with long hair",
            # "woman with short hair",
            # "baby girl",
            # "teenage girl",
            # "girl in her 20s",
            # "girl in her 30s",
            # "middle age woman",
            # "old woman",
            "girl aged 0-2",
            "girl aged 3-9",
            "girl aged 10-19",
            "woman aged 20-29",
            "woman aged 30-39",
            "woman aged 40-49",
            "woman aged 50-59",
            "woman aged 60-69",
            "woman aged more than 70",
        ],
    }
    # augmented_gender_include_labels_dict = {
    #     "Male": [
    #         "man",
    #     ],
    #     "Female": [
    #         "woman",
    #     ]
    # }

    # filter dataset to just be black people
    # dataset = filter_dataset_by_attribute(dataset, labels_dict, "race", ["Black",])
    # with Black Male, Black Female, get 0.89675
    # with Male, Female, get 0.89953
    # with black man, black woman, get 0.89732
    # man, woman: 0.89986
    # man, black woman: 0.87084
    # black man, woman: 0.89037
    # man / old man, woman / old woman: .902

    print(augmented_gender_include_labels_dict)

    # setup augmented classifier run
    augmented_predictions = run_image_classifier_against_ray_dataset(
        dataset,
        batch_size=64,
        labels=gender_labels,
        inc_sub_labels_dict=augmented_gender_include_labels_dict,
    )

    # # compute accuracy
    # augmented_accuracy = augmented_predictions.mean("is_correct")
    # print(f"Accuracy: {augmented_accuracy}")

    # compute accuracy conditional on race
    conditional_gender_accuracy = compute_average_value_conditional_on_attributes(
        augmented_predictions,
        "is_correct",
        labels_dict,
        [
            "age",
            "gender",
        ],
    )
    print(conditional_gender_accuracy)
    # # {'East Asian': 0.9446569545047612, 'Latino_Hispanic': 0.9515972170270068, 'Southeast Asian': 0.9345067160722557, 'Black': 0.8995340472492438, 'Indian': 0.9439077847227859, 'Middle Eastern': 0.9624565972222222, 'White': 0.9587946995825014}
    # # {'Southeast Asian': 0.9345067160722557, 'Latino_Hispanic': 0.9515972170270068, 'Indian': 0.9439077847227859, 'East Asian': 0.9446569545047612, 'Black': 0.8995340472492438, 'Middle Eastern': 0.9624565972222222, 'White': 0.9587946995825014}

    # write_predictions_to_csv(augmented_predictions, "/directai_fastapi/.cache/fairface/black_w_young_fairface_train_pred.csv")

    # {'Black': 0.7490394833646693, 'East Asian': 0.8117522584845772, 'Indian': 0.836512703953243, 'Latino_Hispanic': 0.8446921523154036, 'Middle Eastern': 0.8850911458333334, 'Southeast Asian': 0.8103751736915239, 'White': 0.8602892236945604}
    # {'Black': 0.7580315539932968, 'East Asian': 0.8096362008627004, 'Indian': 0.8335092134101794, 'Latino_Hispanic': 0.8428218747662153, 'Middle Eastern': 0.8845486111111112, 'Southeast Asian': 0.8037980546549328, 'White': 0.8598051673019906}
    # {'Black': 0.752145835036377, 'East Asian': 0.8174493366973223, 'Indian': 0.8313986524880266, 'Latino_Hispanic': 0.8429714969701504, 'Middle Eastern': 0.8858506944444444, 'Southeast Asian': 0.8082445576655859, 'White': 0.8594421250075633}
    # {'Black': 0.7384124908035641, 'East Asian': 0.8322617400504598, 'Indian': 0.7891062586248884, 'Latino_Hispanic': 0.8295803097179621, 'Middle Eastern': 0.8759765625, 'Southeast Asian': 0.8055581287633163, 'White': 0.8546620681309373}
    # {'Black': 0.7576228235101774, 'East Asian': 0.8198909416456417, 'Indian': 0.8326162837892687, 'Latino_Hispanic': 0.843195930276053, 'Middle Eastern': 0.8850911458333334, 'Southeast Asian': 0.8052802223251505, 'White': 0.8596841532038483}
    # {'Black': 0.7576228235101774, 'East Asian': 0.8198909416456417, 'Indian': 0.8326162837892687, 'Latino_Hispanic': 0.843195930276053, 'Middle Eastern': 0.8849826388888888, 'Southeast Asian': 0.8052802223251505, 'White': 0.8596841532038483}

    # {'Black': 0.7599117142156462, 'East Asian': 0.8165540815496053, 'Indian': 0.8411397028979625, 'Latino_Hispanic': 0.8463379965586894, 'Middle Eastern': 0.880859375, 'Southeast Asian': 0.8038906901343215, 'White': 0.8577479276335693}

    # {'Black': 0.8998610316357394, 'East Asian': 0.9422153495564418, 'Indian': 0.9445571880834484, 'Latino_Hispanic': 0.9502506171915912, 'Middle Eastern': 0.9632161458333334, 'Southeast Asian': 0.932561371005095, 'White': 0.9590367277787862}
    # {'Black': 0.8975721409302706, 'East Asian': 0.9468543989582485, 'Indian': 0.9473983277863463, 'Latino_Hispanic': 0.9524949502506171, 'Middle Eastern': 0.9652777777777778, 'Southeast Asian': 0.9386753126447429, 'White': 0.9597628123676408}
    # {'Black': 0.9025586528243277, 'East Asian': 0.9396923577765117, 'Indian': 0.9462618719051871, 'Latino_Hispanic': 0.9509987282112665, 'Middle Eastern': 0.9644097222222222, 'Southeast Asian': 0.9310792033348773, 'White': 0.9594602771222848}

    # big model:
    # w/ old / asian modifiers:
    # {'Black': 0.9023951606310798, 'East Asian': 0.9469357857898592, 'Indian': 0.9486159590875882, 'Latino_Hispanic': 0.9529438168624224, 'Middle Eastern': 0.9654947916666666, 'Southeast Asian': 0.9390458545622974, 'White': 0.9596417982694984, 'Overall_Balanced': 0.945010452409916, 'Overall_Count': 0.9452296412431984}
    # raw:
    # {'Black': 0.8998610316357394, 'East Asian': 0.9422153495564418, 'Indian': 0.9445571880834484, 'Latino_Hispanic': 0.9502506171915912, 'Middle Eastern': 0.9632161458333334, 'Southeast Asian': 0.932561371005095, 'White': 0.9590367277787862, 'Overall_Balanced': 0.9416712044406336, 'Overall_Count': 0.9420478649820161}

    # w/ no modifiers:
    # {('Black', 'Female'): 0.8996252240508392, ('Black', 'Male'): 0.9000984251968503, ('East Asian', 'Female'): 0.9672691744015632, ('East Asian', 'Male'): 0.9171819069313375, ('Indian', 'Female'): 0.9576916567947199, ('Indian', 'Male'): 0.9324492979719189, ('Latino_Hispanic', 'Female'): 0.9626209977661951, ('Latino_Hispanic', 'Male'): 0.9377630787733012, ('Middle Eastern', 'Female'): 0.9525816649104321, ('Middle Eastern', 'Male'): 0.9679698539802166, ('Southeast Asian', 'Female'): 0.9552382789890025, ('Southeast Asian', 'Male'): 0.9116179615110478, ('White', 'Female'): 0.9574495272169691, ('White', 'Male'): 0.9604643144466153, 'Overall_Balanced': 0.9414300973529292, 'Overall_Count': 0.9420478649820161}
    # w/ indian modifier:
    # {('Black', 'Female'): 0.8962033566889359, ('Black', 'Male'): 0.9038713910761155, ('East Asian', 'Female'): 0.9669434945448624, ('East Asian', 'Male'): 0.9178327367393426, ('Indian', 'Female'): 0.9490607547808427, ('Indian', 'Male'): 0.946801872074883, ('Latino_Hispanic', 'Female'): 0.9611317944899479, ('Latino_Hispanic', 'Male'): 0.9424233313289236, ('Middle Eastern', 'Female'): 0.9508254302774851, ('Middle Eastern', 'Male'): 0.9701680012560842, ('Southeast Asian', 'Female'): 0.9533088944626664, ('Southeast Asian', 'Male'): 0.9169636493228795, ('White', 'Female'): 0.9571939688218758, ('White', 'Male'): 0.9603493851281462, 'Overall_Balanced': 0.9423627186423563, 'Overall_Count': 0.9431084570690769}
    # w/ _ from india modifier:
    # {('Black', 'Female'): 0.8939221117810005, ('Black', 'Male'): 0.9063320209973753, ('East Asian', 'Female'): 0.9669434945448624, ('East Asian', 'Male'): 0.9188089814513505, ('Indian', 'Female'): 0.9458453206972415, ('Indian', 'Male'): 0.9488299531981279, ('Latino_Hispanic', 'Female'): 0.9603871928518243, ('Latino_Hispanic', 'Male'): 0.9434756464221287, ('Middle Eastern', 'Female'): 0.9508254302774851, ('Middle Eastern', 'Male'): 0.9701680012560842, ('Southeast Asian', 'Female'): 0.952537140652132, ('Southeast Asian', 'Male'): 0.9198146828225232, ('White', 'Female'): 0.9570661896243292, ('White', 'Male'): 0.9604643144466153, 'Overall_Balanced': 0.9425300343587913, 'Overall_Count': 0.9432813796919671}
    # w/ asian modifier:
    # {('Black', 'Female'): 0.8945738960404106, ('Black', 'Male'): 0.9041994750656168, ('East Asian', 'Female'): 0.9444715844325029, ('East Asian', 'Male'): 0.9497233973315978, ('Indian', 'Female'): 0.947876121171095, ('Indian', 'Male'): 0.9455538221528861, ('Latino_Hispanic', 'Female'): 0.9527922561429635, ('Latino_Hispanic', 'Male'): 0.9503908598917619, ('Middle Eastern', 'Female'): 0.9487179487179487, ('Middle Eastern', 'Male'): 0.9711100643743131, ('Southeast Asian', 'Female'): 0.9245610650202586, ('Southeast Asian', 'Male'): 0.953136136849608, ('White', 'Female'): 0.9551495016611296, ('White', 'Male'): 0.9623031835421216, 'Overall_Balanced': 0.9431828080281582, 'Overall_Count': 0.9441114082818408}
    # w/ southeast asian and asian modifiers:
    # {('Black', 'Female'): 0.8888707837705719, ('Black', 'Male'): 0.9096128608923885, ('East Asian', 'Female'): 0.9446344243608533, ('East Asian', 'Male'): 0.9500488122356003, ('Indian', 'Female'): 0.9411067862582502, ('Indian', 'Male'): 0.9530421216848673, ('Latino_Hispanic', 'Female'): 0.9501116902457185, ('Latino_Hispanic', 'Male'): 0.9530968129885748, ('Middle Eastern', 'Female'): 0.9473129610115911, ('Middle Eastern', 'Male'): 0.9717381064531324, ('Southeast Asian', 'Female'): 0.9247540034728922, ('Southeast Asian', 'Male'): 0.9534925160370634, ('White', 'Female'): 0.9548939432660363, ('White', 'Male'): 0.9624181128605908, 'Overall_Balanced': 0.9432238525384379, 'Overall_Count': 0.9442382182052937}
    # w/ southeast asian and east asian modifiers:
    # {('Black', 'Female'): 0.8888707837705719, ('Black', 'Male'): 0.9096128608923885, ('East Asian', 'Female'): 0.9433317049340498, ('East Asian', 'Male'): 0.9528148389196225, ('Indian', 'Female'): 0.9411067862582502, ('Indian', 'Male'): 0.9528861154446178, ('Latino_Hispanic', 'Female'): 0.9502606105733432, ('Latino_Hispanic', 'Male'): 0.9529464822609741, ('Middle Eastern', 'Female'): 0.9476642079381805, ('Middle Eastern', 'Male'): 0.9718951169728371, ('Southeast Asian', 'Female'): 0.9237893112097241, ('Southeast Asian', 'Male'): 0.9549180327868853, ('White', 'Female'): 0.955021722463583, ('White', 'Male'): 0.9624181128605908, 'Overall_Balanced': 0.9433954776632584, 'Overall_Count': 0.9443996126533247}
    # w/ _ from east asia and southeast asia modifiers:
    # {('Black', 'Female'): 0.8885448916408669, ('Black', 'Male'): 0.9101049868766404, ('East Asian', 'Female'): 0.9425175052922977, ('East Asian', 'Male'): 0.9516758867556134, ('Indian', 'Female'): 0.9411067862582502, ('Indian', 'Male'): 0.9527301092043682, ('Latino_Hispanic', 'Female'): 0.9521965748324647, ('Latino_Hispanic', 'Male'): 0.9521948286229706, ('Middle Eastern', 'Female'): 0.9494204425711275, ('Middle Eastern', 'Male'): 0.9717381064531324, ('Southeast Asian', 'Female'): 0.9232104958518232, ('Southeast Asian', 'Male'): 0.9534925160370634, ('White', 'Female'): 0.9548939432660363, ('White', 'Male'): 0.9623031835421216, 'Overall_Balanced': 0.9432950183717698, 'Overall_Count': 0.9442497463801531}
    # w/ indian, east asian, southeast asian modifiers:
    # {('Black', 'Female'): 0.8885448916408669, ('Black', 'Male'): 0.9094488188976378, ('East Asian', 'Female'): 0.9434945448624003, ('East Asian', 'Male'): 0.9528148389196225, ('Indian', 'Female'): 0.945168387205957, ('Indian', 'Male'): 0.9514820592823713, ('Latino_Hispanic', 'Female'): 0.950409530900968, ('Latino_Hispanic', 'Male'): 0.9527961515333734, ('Middle Eastern', 'Female'): 0.9473129610115911, ('Middle Eastern', 'Male'): 0.9723661485319517, ('Southeast Asian', 'Female'): 0.9245610650202586, ('Southeast Asian', 'Male'): 0.9545616535994298, ('White', 'Female'): 0.955021722463583, ('White', 'Male'): 0.9625330421790599, 'Overall_Balanced': 0.9436082725749336, 'Overall_Count': 0.9446071198007931}
    # w/ indian, east asian, southeast asian, old modifiers:
    # {('Black', 'Female'): 0.9012546846993645, ('Black', 'Male'): 0.9032152230971129, ('East Asian', 'Female'): 0.9464256635727081, ('East Asian', 'Male'): 0.9495606898795965, ('Indian', 'Female'): 0.949399221526485, ('Indian', 'Male'): 0.9502340093603744, ('Latino_Hispanic', 'Female'): 0.9547282204020849, ('Latino_Hispanic', 'Male'): 0.9514431749849669, ('Middle Eastern', 'Female'): 0.9532841587636108, ('Middle Eastern', 'Male'): 0.9712670748940179, ('Southeast Asian', 'Female'): 0.9284198340729307, ('Southeast Asian', 'Male'): 0.9509978617248752, ('White', 'Female'): 0.9589828775875288, ('White', 'Male'): 0.9603493851281462, 'Overall_Balanced': 0.9449687199781288, 'Overall_Count': 0.9457253527621508}
    # w/ middle eastern modifier:
    # {('Black', 'Female'): 0.8999511161805442, ('Black', 'Male'): 0.8997703412073491, ('East Asian', 'Female'): 0.9672691744015632, ('East Asian', 'Male'): 0.9170191994793362, ('Indian', 'Female'): 0.9576916567947199, ('Indian', 'Male'): 0.9335413416536662, ('Latino_Hispanic', 'Female'): 0.963216679076694, ('Latino_Hispanic', 'Male'): 0.9373120865904991, ('Middle Eastern', 'Female'): 0.9536354056902002, ('Middle Eastern', 'Male'): 0.9678128434605119, ('Southeast Asian', 'Female'): 0.9558170943469033, ('Southeast Asian', 'Male'): 0.9110833927298646, ('White', 'Female'): 0.9577050856120624, ('White', 'Male'): 0.9604643144466153, 'Overall_Balanced': 0.941592123690752, 'Overall_Count': 0.9421746749054689}

    # w/ no modifiers:
    # {('0-2', 'Female'): 0.8583815028901735, ('0-2', 'Male'): 0.7009090909090909, ('10-19', 'Female'): 0.9417650590354213, ('10-19', 'Male'): 0.8453482708231855, ('20-29', 'Female'): 0.9704129566009956, ('20-29', 'Male'): 0.9658579620644023, ('3-9', 'Female'): 0.9273550334123734, ('3-9', 'Male'): 0.7682440630958571, ('30-39', 'Female'): 0.9585510688836104, ('30-39', 'Male'): 0.9844875346260388, ('40-49', 'Female'): 0.9509081983308787, ('40-49', 'Male'): 0.9874062968515742, ('50-59', 'Female'): 0.9320875113947129, ('50-59', 'Male'): 0.993554784333168, ('60-69', 'Female'): 0.8798076923076923, ('60-69', 'Male'): 0.9821736630247269, ('more than 70', 'Female'): 0.8929384965831435, ('more than 70', 'Male'): 0.967741935483871, 'Overall_Balanced': 0.9171072844806064, 'Overall_Count': 0.9420363368071567}
    # w/ baby modifier
    # {('0-2', 'Female'): 0.7875722543352601, ('0-2', 'Male'): 0.7972727272727272, ('10-19', 'Female'): 0.929957974784871, ('10-19', 'Male'): 0.8711641500243547, ('20-29', 'Female'): 0.969431395919512, ('20-29', 'Male'): 0.9661226290251433, ('3-9', 'Female'): 0.898038370338435, ('3-9', 'Male'): 0.8387935517420697, ('30-39', 'Female'): 0.9576009501187649, ('30-39', 'Male'): 0.9843951985226224, ('40-49', 'Female'): 0.9501718213058419, ('40-49', 'Male'): 0.9874062968515742, ('50-59', 'Female'): 0.9316317228805834, ('50-59', 'Male'): 0.993554784333168, ('60-69', 'Female'): 0.8788461538461538, ('60-69', 'Male'): 0.9821736630247269, ('more than 70', 'Female'): 0.8906605922551253, ('more than 70', 'Male'): 0.967741935483871, 'Overall_Balanced': 0.9212520095591559, 'Overall_Count': 0.946059669833072}
    # w/ baby, teenage modifiers
    # {('0-2', 'Female'): 0.7875722543352601, ('0-2', 'Male'): 0.7990909090909091, ('10-19', 'Female'): 0.8845307184310587, ('10-19', 'Male'): 0.9305893813930833, ('20-29', 'Female'): 0.9537965364930239, ('20-29', 'Male'): 0.9739744155271284, ('3-9', 'Female'): 0.8728174175468851, ('3-9', 'Male'): 0.8779684520714162, ('30-39', 'Female'): 0.9460807600950119, ('30-39', 'Male'): 0.9860572483841182, ('40-49', 'Female'): 0.9447717231222386, ('40-49', 'Male'): 0.9874062968515742, ('50-59', 'Female'): 0.9302643573381951, ('50-59', 'Male'): 0.993554784333168, ('60-69', 'Female'): 0.8788461538461538, ('60-69', 'Male'): 0.9821736630247269, ('more than 70', 'Female'): 0.8906605922551253, ('more than 70', 'Male'): 0.967741935483871, 'Overall_Balanced': 0.9215498666457194, 'Overall_Count': 0.9447915705985428}
    # w/ baby, old modifiers
    # {('0-2', 'Female'): 0.7875722543352601, ('0-2', 'Male'): 0.7963636363636364, ('10-19', 'Female'): 0.9305583350010006, ('10-19', 'Male'): 0.8692157817827569, ('20-29', 'Female'): 0.9701325106920002, ('20-29', 'Male'): 0.9649757388619321, ('3-9', 'Female'): 0.8989006251347273, ('3-9', 'Male'): 0.8374068296065176, ('30-39', 'Female'): 0.9610451306413301, ('30-39', 'Male'): 0.9824561403508771, ('40-49', 'Female'): 0.9597447226313206, ('40-49', 'Male'): 0.9824587706146927, ('50-59', 'Female'): 0.9612579762989972, ('50-59', 'Male'): 0.9831432821021319, ('60-69', 'Female'): 0.9384615384615385, ('60-69', 'Male'): 0.9695227142035653, ('more than 70', 'Female'): 0.9498861047835991, ('more than 70', 'Male'): 0.9305210918114144, 'Overall_Balanced': 0.9263123990931833, 'Overall_Count': 0.9469242829475237}

    # w/ baby, old modifiers
    # {('Black', 'Female'): 0.9009287925696594, ('Black', 'Male'): 0.9145341207349081, ('East Asian', 'Female'): 0.9631981761928025, ('East Asian', 'Male'): 0.9272697689554181, ('Indian', 'Female'): 0.9585378236588256, ('Indian', 'Male'): 0.9425897035881435, ('Latino_Hispanic', 'Female'): 0.9597915115413254, ('Latino_Hispanic', 'Male'): 0.9487372218881539, ('Middle Eastern', 'Female'): 0.9532841587636108, ('Middle Eastern', 'Male'): 0.9742502747684095, ('Southeast Asian', 'Female'): 0.9544665251784681, ('Southeast Asian', 'Male'): 0.925160370634355, ('White', 'Female'): 0.9571939688218758, ('White', 'Male'): 0.9665555683254798, 'Overall_Balanced': 0.9461784275443883, 'Overall_Count': 0.9469127547726643}
    # w/ baby, old, east asian, southeast asian modifiers
    # {('Black', 'Female'): 0.894085057845853, ('Black', 'Male'): 0.9189632545931758, ('East Asian', 'Female'): 0.9467513434294089, ('East Asian', 'Male'): 0.9467946631955744, ('Indian', 'Female'): 0.9473684210526315, ('Indian', 'Male'): 0.9502340093603744, ('Latino_Hispanic', 'Female'): 0.9517498138495905, ('Latino_Hispanic', 'Male'): 0.9549007817197835, ('Middle Eastern', 'Female'): 0.9487179487179487, ('Middle Eastern', 'Male'): 0.9756633694457528, ('Southeast Asian', 'Female'): 0.9293845263360988, ('Southeast Asian', 'Male'): 0.9488595866001426, ('White', 'Female'): 0.9554050600562228, ('White', 'Male'): 0.9677048615101712, 'Overall_Balanced': 0.9454701926937664, 'Overall_Count': 0.9465553813520243}
    # w/ baby, old, east asian, southeast asian, black modifiers
    # {('Black', 'Female'): 0.9097278800716962, ('Black', 'Male'): 0.8987860892388452, ('East Asian', 'Female'): 0.9467513434294089, ('East Asian', 'Male'): 0.9464692482915718, ('Indian', 'Female'): 0.9482145879167372, ('Indian', 'Male'): 0.9486739469578783, ('Latino_Hispanic', 'Female'): 0.9524944154877141, ('Latino_Hispanic', 'Male'): 0.9541491280817799, ('Middle Eastern', 'Female'): 0.9487179487179487, ('Middle Eastern', 'Male'): 0.9755063589260481, ('Southeast Asian', 'Female'): 0.9295774647887324, ('Southeast Asian', 'Male'): 0.9485032074126871, ('White', 'Female'): 0.9554050600562228, ('White', 'Male'): 0.9677048615101712, 'Overall_Balanced': 0.94504868149196, 'Overall_Count': 0.9461403670570875}

    # w/ 0-2, 3-9, etc etc
    # {('Black', 'Female'): 0.8680136874694476, ('Black', 'Male'): 0.9440616797900262, ('East Asian', 'Female'): 0.9522879009933236, ('East Asian', 'Male'): 0.9397982427595184, ('Indian', 'Female'): 0.9421221864951769, ('Indian', 'Male'): 0.9600624024960999, ('Latino_Hispanic', 'Female'): 0.951451973194341, ('Latino_Hispanic', 'Male'): 0.9607636800962117, ('Middle Eastern', 'Female'): 0.9438004917456972, ('Middle Eastern', 'Male'): 0.9794316219186685, ('Southeast Asian', 'Female'): 0.9365232490835423, ('Southeast Asian', 'Male'): 0.9428011404133999, ('White', 'Female'): 0.9524661385126502, ('White', 'Male'): 0.9723020342489369, 'Overall_Balanced': 0.9461347449440741, 'Overall_Count': 0.947512219865351}
    # w/ 0-2, 3-9, etc etc
    # {('0-2', 'Female'): 0.7485549132947977, ('0-2', 'Male'): 0.8463636363636363, ('10-19', 'Female'): 0.8917350410246148, ('10-19', 'Male'): 0.9227959084266927, ('20-29', 'Female'): 0.9600364579681694, ('20-29', 'Male'): 0.9690339655932951, ('3-9', 'Female'): 0.8635481784867428, ('3-9', 'Male'): 0.8892355694227769, ('30-39', 'Female'): 0.9576009501187649, ('30-39', 'Male'): 0.9811634349030471, ('40-49', 'Female'): 0.9604810996563574, ('40-49', 'Male'): 0.9823088455772114, ('50-59', 'Female'): 0.95897903372835, ('50-59', 'Male'): 0.9871095686663361, ('60-69', 'Female'): 0.926923076923077, ('60-69', 'Male'): 0.9746981023576768, ('more than 70', 'Female'): 0.9316628701594533, ('more than 70', 'Male'): 0.9330024813895782, 'Overall_Balanced': 0.9269573963366986, 'Overall_Count': 0.947512219865351}
    # w/ 0-2, 3-9, etc etc, but not man / woman
    # {('0-2', 'Female'): 0.7485549132947977, ('0-2', 'Male'): 0.8454545454545455, ('10-19', 'Female'): 0.8913348008805283, ('10-19', 'Male'): 0.9235265465172917, ('20-29', 'Female'): 0.9598261235364229, ('20-29', 'Male'): 0.968504631671813, ('3-9', 'Female'): 0.8629014873895237, ('3-9', 'Male'): 0.889755590223609, ('30-39', 'Female'): 0.9590261282660333, ('30-39', 'Male'): 0.979870729455217, ('40-49', 'Female'): 0.9626902307314679, ('40-49', 'Male'): 0.9806596701649175, ('50-59', 'Female'): 0.9612579762989972, ('50-59', 'Male'): 0.9858701041150223, ('60-69', 'Female'): 0.926923076923077, ('60-69', 'Male'): 0.9741230592294422, ('more than 70', 'Female'): 0.9316628701594533, ('more than 70', 'Male'): 0.9330024813895782, 'Overall_Balanced': 0.9269413869834298, 'Overall_Count': 0.94735082541732}

    # ON VALIDATION SET
    # (this is our only run on the validation set)
    # w/ no modifiers:
    # {('0-2', 'Female'): 0.881578947368421, ('0-2', 'Male'): 0.7560975609756098, ('10-19', 'Female'): 0.9240506329113924, ('10-19', 'Male'): 0.8378870673952641, ('20-29', 'Female'): 0.9705240174672489, ('20-29', 'Male'): 0.9598092643051771, ('3-9', 'Female'): 0.9219512195121952, ('3-9', 'Male'): 0.7989203778677463, ('30-39', 'Female'): 0.9656188605108055, ('30-39', 'Male'): 0.9817073170731707, ('40-49', 'Female'): 0.9439071566731141, ('40-49', 'Male'): 0.9868421052631579, ('50-59', 'Female'): 0.92, ('50-59', 'Male'): 0.9858870967741935, ('60-69', 'Female'): 0.8596491228070176, ('60-69', 'Male'): 0.9903381642512077, ('more than 70', 'Female'): 0.9137931034482759, ('more than 70', 'Male'): 0.9666666666666667, 'Overall_Balanced': 0.9202904822928146, 'Overall_Count': 0.9414825634471427}
    # w/ 0-2, 3-9, etc etc
    # {('0-2', 'Female'): 0.7894736842105263, ('0-2', 'Male'): 0.8943089430894309, ('10-19', 'Female'): 0.870253164556962, ('10-19', 'Male'): 0.912568306010929, ('20-29', 'Female'): 0.9585152838427947, ('20-29', 'Male'): 0.9604904632152589, ('3-9', 'Female'): 0.8552845528455284, ('3-9', 'Male'): 0.9001349527665317, ('30-39', 'Female'): 0.9577603143418467, ('30-39', 'Male'): 0.9809451219512195, ('40-49', 'Female'): 0.9516441005802708, ('40-49', 'Male'): 0.9820574162679426, ('50-59', 'Female'): 0.95, ('50-59', 'Male'): 0.9798387096774194, ('60-69', 'Female'): 0.8947368421052632, ('60-69', 'Male'): 0.9710144927536232, ('more than 70', 'Female'): 0.9482758620689655, ('more than 70', 'Male'): 0.95, 'Overall_Balanced': 0.9281834561269174, 'Overall_Count': 0.9440387073215264}
