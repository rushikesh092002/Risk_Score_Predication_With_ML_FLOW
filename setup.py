import setuptools

with open("README.md" , "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Risk_Score_Predication_With_ML_FLOW"
AUTHOR_USER_NAME = "rushikesh092002"
SRC_REPO = "RiskScorePrediction"
AUTHOR_EMAIL = "rushikeshgaikhe09@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A machine learning project to predict the risk score of a borrower",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls ={
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src")
)