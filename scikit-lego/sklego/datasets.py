def load_chicken(return_X_y=False, as_frame=False):
    """Loads the chicken dataset.

    The chicken data has 578 rows and 4 columns from an experiment on the effect of diet on early growth of chicks.
    The body weights of the chicks were measured at birth and every second day thereafter until day 20.
    They were also measured on day 21. There were four groups on chicks on different protein diets.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    -------
    ```py
    from sklego.datasets import load_chicken
    X, y = load_chicken(return_X_y=True)

    X.shape
    # (578, 3)
    y.shape
    # (578,)

    load_chicken(as_frame=True).columns
    # Index(['weight', 'time', 'chick', 'diet'], dtype='object')
    ```

    The datasets can be found in the following sources:

    - Crowder, M. and Hand, D. (1990), Analysis of Repeated Measures, Chapman and Hall (example 5.3)
    - Hand, D. and Crowder, M. (1996), Practical Longitudinal Data Analysis, Chapman and Hall (table A.2)
    """
    filepath = importlib_resources.files("sklego") / "data" / "chickweight.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = df[["time", "diet", "chick"]].to_numpy(), df["weight"].to_numpy()
    if return_X_y:
        return X, y
    return {"data": X, "target": y}

def load_penguins(return_X_y=False, as_frame=False):
    """Loads the penguins dataset, which is a lovely alternative for the iris dataset.
    We've added this dataset for educational use.

    Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of
    the Long Term Ecological Research Network. The goal of the dataset is to predict which species of penguin a penguin
    belongs to.

    This data originally appeared as a R package and R users can find this data in the
    [palmerpenguins package](https://github.com/allisonhorst/palmerpenguins).
    You can also go to the repository for some lovely images that explain the dataset.

    To cite this dataset in publications use:

        Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism
        and Environmental Variability within a Community of Antarctic
        Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081.
        https://doi.org/10.1371/journal.pone.0090081

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    -------
    ```py
    from sklego.datasets import load_penguins
    X, y = load_penguins(return_X_y=True)

    X.shape
    # (344, 6)
    y.shape
    # (344,)

    load_penguins(as_frame=True).columns
    # Index(['species', 'island', 'bill_length_mm', 'bill_depth_mm',
    #    'flipper_length_mm', 'body_mass_g', 'sex'],
    #   dtype='object')
    ```

    Notes
    -----
    Anyone interested in publishing the data should contact
    [`Dr. Kristen Gorman`](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)
    about analysis and working together on any final products.

    !!! quote "From Gorman et al. (2014)"

        Data reported here are publicly available within the
        [PAL-LTER data system (datasets 219, 220, and 221)](http://oceaninformatics.ucsd.edu/datazoo/data/pallter/datasets).

        Individuals interested in using these data are therefore expected to follow the [US LTER Network's Data Access
        Policy, Requirements and Use Agreement](https://lternet.edu/data-access-policy/)

    Please cite data using the following
    ------------------------------------
    **Adélie penguins:**

      - [Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size measurements and isotopic signatures of
        foraging among adult male and female Adélie penguins (*Pygoscelis adeliae*) nesting along the Palmer Archipelago
        near Palmer Station, 2007-2009 ver 5. Environmental Data
        Initiative](https://doi.org/10.6073/pasta/98b16d7d563f265cb52372c8ca99e60f).

        (Accessed 2020-06-08).

    **Gentoo penguins:**

      - [Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size measurements and isotopic signatures of
        foraging among adult male and female Gentoo penguin (*Pygoscelis papua*) nesting along the Palmer Archipelago
        near Palmer Station, 2007-2009 ver 5. Environmental Data
        Initiative](https://doi.org/10.6073/pasta/7fca67fb28d56ee2ffa3d9370ebda689).

        (Accessed 2020-06-08).

    **Chinstrap penguins:**

      - [Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size measurements and isotopic signatures of
        foraging among adult male and female Chinstrap penguin (*Pygoscelis antarcticus*) nesting along the Palmer
        Archipelago near Palmer Station, 2007-2009 ver 6.
        Environmental Data Initiative](https://doi.org/10.6073/pasta/c14dfcfada8ea13a17536e73eb6fbe9e).

        (Accessed 2020-06-08).
    """
    filepath = importlib_resources.files("sklego") / "data" / "penguins.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[
            [
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
            ]
        ].to_numpy(),
        df["species"].to_numpy(),
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}

def load_abalone(return_X_y=False, as_frame=False):
    """
    Loads the abalone dataset where the goal is to predict the gender of the creature.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    ---------
    ```py
    from sklego.datasets import load_abalone
    X, y = load_abalone(return_X_y=True)

    X.shape
    # (4177, 8)
    y.shape
    # (4177,)

    load_abalone(as_frame=True).columns
    # Index(['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
    #        'viscera_weight', 'shell_weight', 'rings'],
    #       dtype='object')
    ```

    The dataset was copied from Kaggle and can originally be found in the following sources:

    - Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994)

        "The Population Biology of Abalone (_Haliotis_ species) in Tasmania."

        Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288)
    """
    filepath = importlib_resources.files("sklego") / "data" / "abalone.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X = df[
        [
            "length",
            "diameter",
            "height",
            "whole_weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight",
            "rings",
        ]
    ].to_numpy()
    y = df["sex"].to_numpy()
    if return_X_y:
        return X, y
    return {"data": X, "target": y}

