import argostranslate.package
import argostranslate.translate
from googletrans import Translator

available_from_code = [
    x.from_code for x in argostranslate.package.get_available_packages()
]
available_to_code = [x.to_code for x in argostranslate.package.get_available_packages()]
# ensure that the target language is available both as source and target
available_code = list(set(available_from_code + available_to_code))


def install_package(from_code, to_code):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages,
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())


def check_language_availability(from_code, to_code):
    installed_package = argostranslate.package.get_installed_packages()
    installed_from_code = [x.from_code for x in installed_package]
    installed_to_code = [x.to_code for x in installed_package]
    if not (from_code in installed_from_code and to_code in installed_to_code):
        if from_code not in available_code or to_code not in available_code:
            raise ValueError(f"Language {from_code} not available")
        install_package(from_code, to_code)
        install_package(to_code, from_code)


class TranslateBackAugment(object):
    def __init__(self, src, to):
        check_language_availability(src, to)
        self.src = src
        self.to = to

    def augment(self, data):
        if type(data) is not str:
            raise TypeError("DataType must be a string")
        try:
            data = data.lower()
            data = argostranslate.translate.translate(data, self.src, self.to)
            data = argostranslate.translate.translate(data, self.to, self.src)
        except Exception:
            try:  # Switch to googletrans to do translation.
                translator = Translator()
                data = translator.translate(data, dest=self.to, src=self.src).text
                data = translator.translate(data, dest=self.src, src=self.to).text
            except Exception:
                print("Error Not translated.\n")
                raise

        return str(data).lower()


if __name__ == "__main__":
    tb = TranslateBackAugment(src="en", to="de")
    print(tb.augment("I love to go to school every day!"))
