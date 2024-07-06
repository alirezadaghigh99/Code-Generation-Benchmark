def parse_label_loss(loss_config: Dict) -> Dict:
    """
    Parse the label loss part in loss configuration.

    :param loss_config: potentially outdated config
    :return: latest config
    """
    if "label" not in loss_config:
        # no label loss
        return loss_config

    if isinstance(loss_config["label"], list):
        # config up-to-date
        return loss_config

    label_loss_name = loss_config["label"]["name"]
    if label_loss_name == "single_scale":
        loss_config["label"] = {
            "name": loss_config["label"]["single_scale"]["loss_type"],
            "weight": loss_config["label"].get("weight", 1.0),
        }
    elif label_loss_name == "multi_scale":
        loss_config["label"] = {
            "name": loss_config["label"]["multi_scale"]["loss_type"],
            "weight": loss_config["label"].get("weight", 1.0),
            "scales": loss_config["label"]["multi_scale"]["loss_scales"],
        }

    # mean-squared renamed to ssd
    if loss_config["label"]["name"] == "mean-squared":
        loss_config["label"]["name"] = "ssd"

    # dice_generalized merged into dice
    if loss_config["label"]["name"] == "dice_generalized":
        loss_config["label"]["name"] = "dice"

    # rename neg_weight to background_weight
    if "neg_weight" in loss_config["label"]:
        background_weight = loss_config["label"].pop("neg_weight")
        loss_config["label"]["background_weight"] = background_weight

    return loss_config

def parse_reg_loss(loss_config: Dict) -> Dict:
    """
    Parse the regularization loss part in loss configuration.

    :param loss_config: potentially outdated config
    :return: latest config
    """
    if "regularization" not in loss_config:
        # no regularization loss
        return loss_config

    if isinstance(loss_config["regularization"], list):
        # config up-to-date
        return loss_config

    if "energy_type" not in loss_config["regularization"]:
        # up-to-date
        return loss_config

    energy_type = loss_config["regularization"]["energy_type"]
    reg_config = {"weight": loss_config["regularization"].get("weight", 1.0)}
    if energy_type == "bending":
        reg_config["name"] = "bending"
    elif energy_type == "gradient-l2":
        reg_config["name"] = "gradient"
        reg_config["l1"] = False
    elif energy_type == "gradient-l1":
        reg_config["name"] = "gradient"
        reg_config["l1"] = True
    loss_config["regularization"] = reg_config

    return loss_config

