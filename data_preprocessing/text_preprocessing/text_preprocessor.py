import re
from typing import Dict, List, Tuple
import pandas as pd
import random
import language_tool_python
import textwrap
from concurrent.futures import ThreadPoolExecutor
import tqdm
import nltk
from nltk.corpus import stopwords


class DataHandler:
    def __init__(
        self, medical_reports_per_patient, use_heuristics=False, verbose=False
    ):
        self.medical_reports_per_patient = medical_reports_per_patient
        self.use_heuristics = use_heuristics
        self.verbose = verbose

    def flatten_dict(self, nested_dict, level):
        """
        Function to flatten a nested dictionary. This is done to have a more
        structured way of working with the data. The nested dictionary is
        flattened to a dictionary with a tuple as key. In level 2, the tuple
        contains the patient and the report number. In level 3, the tuple
        contains the patient, the report number and the section.

        Args:
        nested_dict: A nested dictionary containing the medical reports
        level: The level to which the dictionary should be flattened.
            level 2: (patient, report_nr)
            level 3: (patient, report_nr, section)

        Returns:
        Dict: A dictionary containing the flattened reports
        """
        flattened_dict = {}

        if level == 2:
            for patient, reports in nested_dict.items():
                for report_nr, report in reports.items():
                    flattened_dict[(patient, report_nr)] = report

        elif level == 3:
            for patient, reports in nested_dict.items():
                for report_nr, report in reports.items():
                    for section, content in report.items():
                        flattened_dict[(patient, report_nr, section)] = content

        return flattened_dict

    def unflatten_dict(self, flat_dict, level):
        """
        Function to unflatten a dictionary. This is done to have a more
        structured way of working with the data. The dictionary is unflattened
        to a nested dictionary. In level 2, the tuple contains the patient and
        the report number. In level 3, the tuple contains the patient, the
        report number and the section.

        Args:
        flat_dict: A dictionary containing the flattened reports
        level: The level to which the dictionary should be unflattened.
            level 2: (patient, report_nr)
            level 3: (patient, report_nr, section)

        Returns:
        Dict: A dictionary containing the unflattened reports
        """
        unflattened_dict = {}

        if level == 2:
            for (patient, report_nr), content in flat_dict.items():
                unflattened_dict.setdefault(patient, {})[report_nr] = content

        elif level == 3:
            for (patient, report_nr, section), content in flat_dict.items():
                unflattened_dict.setdefault(patient, {}).setdefault(report_nr, {})[
                    section
                ] = content

        return unflattened_dict

    def join_sentences_with_space(self, list_of_sentences: List[str]) -> str:
        return " ".join(list_of_sentences)

    def structure_section_preprocessing(
        self, structure_report_sentences: List[str]
    ) -> str:
        """
        Preprocess the `structured_report_en` section. This is independently
        preprocessed because, unlike the other sections, this section is
        written in a line-based structure instead of a sentence-based
        structure. This means that the sentences are seperated by a newline
        character whereas the other sections are seperated by a period.

        The preprocessing steps that are done are:
        1) Find if there are any sentences that are split by a newline
            character
        2) Replace the newline character with a period
        3) Join the sentences together to create a single string

        Args:
        structure_report_sentences (List[str]): A list of sentences that are
            part of the `structured_report_en` section

        Returns:
        str: a single string that contains all the sentences of the
            `structured_report_en` section joined together as a piece of text
        """
        # Combine all sentences into one string
        text = ". ".join(structure_report_sentences)
        # Replace newline characters (\n) and literal \n with '. '
        text = re.sub(r"(\\n|\n)+", ". ", text)
        # Remove any extra spaces
        text = re.sub(r"\s+", " ", text)
        # Replace ';' with '.'
        text = re.sub(r";", ".", text)
        return text.strip()

    def preprocess_each_section(
        self, flat_dict: Dict[Tuple[str, str, str], List[str]]
    ) -> Dict[Tuple[str, str, str], str]:
        """
        Function to join the sentences of each section together. This is done
        to create a single string for each section. For `structured_report_en`
        a different preprocessing step is performed.

        Args:
        flat_dict (Dict[Tuple[str, str, str], List[str]]): A dictionary
            containing the flattened reports

        Returns:
        Dict[Tuple[str, str, str], str]: A dictionary containing the
            preprocessed reports with the sections joined together
        """
        new_flat_dict = {}
        for key, sentences in flat_dict.items():
            patient, report_nr, section = key
            if section == "structured_report_en":
                new_flat_dict[key] = self.structure_section_preprocessing(sentences)
            elif section == "conclusion_en" and self.use_heuristics:
                # All sentences that have given in the conclusion are removed,
                # this has been checked manually and is a heuristic
                new_flat_dict[key] = self.join_sentences_with_space(
                    [
                        re.sub(r";", ".", sentence)
                        for sentence in sentences
                        if "given" not in sentence.lower()
                    ]
                )
            else:
                new_flat_dict[key] = self.join_sentences_with_space(sentences)

        return new_flat_dict

    def delete_duplicate_sentences(self, text: str) -> Tuple[str, int]:
        """
        Delete the latter repeated sentences in a text.

        Assumptions:
        - Text is a string, where sentences are separated by a period (".").
        - If a sentence is repeated, the latter one is deleted.

        Args:
        text (str): A string of sentences.

        Returns:
        str: A string of sentences without the repeated ones.
        int: The number of repeated sentences.
        """
        sentences = re.split(r"\.\s*", text)
        seen = set()
        new_sentences = []
        duplicates = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence != "") and (sentence not in seen):
                seen.add(sentence)
                new_sentences.append(sentence)
            elif sentence == "":
                pass
            else:
                duplicates += 1
                if self.verbose:
                    print(f"Removed duplicate sentence: {sentence}")

        text = ". ".join(new_sentences)
        try:
            if text[-1] != ".":
                if text[-1] == ",":
                    text = text[:-1] + "."
                else:
                    text += "."
        except (IndexError, AttributeError):
            pass

        return text, duplicates

    def join_sections(
        self, preprocessed_reports_per_section: Dict[str, Dict[str, Dict[str, str]]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Join the sections together. This is done to create a single string for
        each report.

        Args:
        preprocessed_reports_per_section
            (Dict[str, Dict[str, Dict[str, str]]]): A dictionary containing the
            preprocessed reports per section

        Returns:
        Dict[str, Dict[str, str]]: A dictionary containing the preprocessed
            reports with the sections joined together
        """
        joined_reports = {}
        for patient, reports in preprocessed_reports_per_section.items():
            joined_reports[patient] = {}
            for report_nr, report in reports.items():
                section_content = []
                for content in report.values():
                    # Check if the last character is a period
                    content = content.strip()
                    if content and content[-1] != ".":
                        content += "."
                    section_content.append(content)
                joined_reports[patient][report_nr] = " ".join(section_content)
        return joined_reports


class ReportAnalyzer:
    def __init__(self, medical_reports_per_patient):
        self.medical_reports_per_patient = medical_reports_per_patient

    def analyze_sentences_per_section(self) -> pd.DataFrame:
        """
        Analyze the number of sentences per section. We assume that the
        sentences are already seperated or that the sentences are seperated by
        a '.' or '\\n'. It is used to detect if certain sentences are repeated
        very often that are not relevant and thus could be removed.

        Returns:
        info_per_section (pd.DataFrame): A DataFrame containing all sentences
            with the amount of occurences per section
        """
        sentences = {}
        for patient, medical_reports in self.medical_reports_per_patient.items():
            for report_nr, report in medical_reports.items():
                for section, content in report.items():
                    if section not in sentences:
                        sentences[section] = {}
                    for sentence in content:
                        # Regex pattern to split on '.' and '\\n' but keep the
                        # delimiter
                        splitted_parts = re.split(r"(?<=\.) |(?<=\\n)", sentence)
                        for part in splitted_parts:
                            part = part.strip()
                            if part:
                                if part not in sentences[section]:
                                    sentences[section][part] = 0
                                sentences[section][part] += 1
        return pd.DataFrame(sentences)

    def information_about_medical_reports(self) -> pd.DataFrame:
        """
        Create a dataframe that contains information about the medical
        reports. The information that is included is the existence of certain
        sections and the patient that the report belongs to. This dataframe is
        used to analyze the structure of the medical reports and to trace back
        the patient that the report belongs to.

        Returns:
        report_information (pd.DataFrame): A DataFrame containing information
            about the medical reports
        """
        medical_reports_info = {}
        for patient, medical_reports in self.medical_reports_per_patient.items():
            for report_nr, report in medical_reports.items():
                medical_reports_info[report_nr] = {}
                for section, content in report.items():
                    if section not in medical_reports_info[report_nr]:
                        medical_reports_info[report_nr][section] = int(1)
                medical_reports_info[report_nr]["patient"] = str(patient)
        df = pd.DataFrame(medical_reports_info).T.fillna(0)
        return df[
            [
                "structured_report_en",
                "description_en",
                "discussion_en",
                "conclusion_en",
                "patient",
            ]
        ]

    def capital_word_analysis(
        self, flattened_reports: Dict[Tuple[str, str, str], str]
    ) -> List[str]:
        """
        Analyze the words that start with a capital letter and are not the
        first word of a sentence. This is done to filter out the words that are
        not relevant and are able to be filtered out.

        Args:
        flattened_reports (Dict[Tuple[str, str, str], str]): A dictionary
            containing the reports that are flattened

        Returns:
        List[str]: A list containing the words that are able to be filtered out
        """
        capital_words = set()

        # Collect all words that start with a capital letter and are not the
        # first word of a sentence
        for key, content in flattened_reports.items():
            sentences = re.split(r"(?<=[.!?])\s+", content)
            for sentence in sentences:
                words = re.findall(r"\b[\w&-]+\b", sentence)
                for idx, word in enumerate(words):
                    if word[0].isupper() and idx != 0:
                        capital_words.add(word)

        # Download stopwords which are able to be filtered out
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

        # After investigating `capital_words`, we can filter out the words that
        # are not relevant
        manual_words = [
            "Centrally",
            "Furthermore",
            "Also",
            "Focally",
            "Given",
            "Locally",
            "However",
            "Especially",
            "Towards",
            "Additionally",
            "Here",
            "Due",
            "Despite",
            "Although",
            "Therefore",
            "Thus",
            "Microsatellites",
        ]

        # Aggregated words that are filtered out
        filtered_words = stop_words.union([word.lower() for word in manual_words])

        # Filter out the words that are not relevant
        capital_words_filtered = [
            word for word in list(capital_words) if word.lower() not in filtered_words
        ]

        return capital_words_filtered


class GrammarChecker:
    def __init__(self, num_servers, is_good_rule, verbose=False):
        self.num_servers = num_servers
        self.is_good_rule = is_good_rule
        self.verbose = verbose
        self.servers = []
        self.mistake_bool_dict = {}

    def initialize_mistake_bool_dict(self, flat_medical_reports) -> Dict[tuple, bool]:
        """
        Initialize a dictionary that contains if a medical report (section)
        needs to be checked for mistakes. This is done to keep filter out
        the reports that do not need to be checked.

        Args:
        flat_medical_reports (Shape is not fixed): A dictionary containing the
            preprocessed reports that are flattened

        Returns:
        Dict[tuple, bool]: A dictionary containing if a something needs to be
            checked for mistakes
        """
        mistake_bool_dict = {}
        for key in flat_medical_reports.keys():
            mistake_bool_dict[key] = True
        return mistake_bool_dict

    def initialize_language_tools(self):
        self.servers = [
            language_tool_python.LanguageTool("en-US")
            for _ in tqdm.tqdm(range(self.num_servers), desc="Initializing Servers")
        ]

    def close_language_tools(self):
        if self.servers:
            for server in self.servers:
                server.close()

    def grammar_check(self, key, content, tool):
        """
        Use the `language_tool_python` library to check grammar/structure/
        punctuation/etc. mistakes in the text. Only the rules that are
        considered to be good are adjusted automatically. These rules are
        chosen by the `is_good_rule` function and are defined by ourselves.

        Args:
        key (tuple): The key that is used to identify the content
        content (str): The content that should be checked for grammar mistakes
        tool (language_tool_python.LanguageTool): The tool that is used to
            check the content

        Returns:
        str: The (corrected) content
        """
        matches = tool.check(content)

        if matches:
            matches = [rule for rule in matches if self.is_good_rule(rule)]
            if len(matches) == 0:
                self.mistake_bool_dict[key] = False
                return content
            corrected = language_tool_python.utils.correct(content, matches)

            # if self.verbose:
            #     print(f"Key: {key}")
            #     print(f"Original report:  {content}")
            #     print(f"Corrected report: {corrected}")
            #     print("")

            return corrected
        else:
            self.mistake_bool_dict[key] = False
            return content

    def grammar_check_report(self, reports):
        """
        Check the grammar of the reports. This is done by using the
        `language_tool_python` library. The grammar is checked recursively
        until no mistakes are found in the reports. The grammar check is
        performed using ThreadPoolExecutor to speed up the process.

        Args:
        preprocessed_reports_per_section
            (Dict[str, Dict[str, Dict[str, str]]]): A dictionary containing the
              preprocessed reports per section

        Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing the
            preprocessed reports with the grammar mistakes corrected
        """

        # Create dictionary for filtering reports without mistakes
        self.mistake_bool_dict = self.initialize_mistake_bool_dict(reports)

        # Initialize language tools
        self.initialize_language_tools()

        # Check grammar for each report
        iteration = 0
        while True:
            # Identify reports with mistakes
            reports_with_mistakes = [
                key for key, value in self.mistake_bool_dict.items() if value
            ]

            if len(reports_with_mistakes) == 0:
                print("No mistakes left in the reports")
                break

            if iteration == 0:
                print(
                    f"Amount of reports/report sections to check: "
                    f"{len(reports_with_mistakes)}"
                )
            else:
                print(
                    f"Amount of reports/report sections left to check after "
                    f"iteration {iteration}: {len(reports_with_mistakes)}"
                )

            tasks = [
                (key, reports[key], self.servers[i % len(self.servers)])
                for i, key in enumerate(reports_with_mistakes)
            ]

            def process_report(args):
                key, content, tool = args
                corrected_content = self.grammar_check(key, content, tool)
                return key, corrected_content

            with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
                results = list(
                    tqdm.tqdm(
                        executor.map(process_report, tasks),
                        total=len(tasks),
                        desc="Checking Reports",
                    )
                )

            for key, corrected_report in results:
                reports[key] = corrected_report

            iteration += 1

            if iteration >= 9:
                print("Reached maximum amount of iterations")
                break

        # Close language tools
        self.close_language_tools()

        return reports


class HeuristicsProcessor:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.words_after = set()

    def capital_word_adjustment(self, reports, capital_words_filtered):
        """
        If you observe a capital word in the middle of a sentence, and the word
        is not in the list of capital words that are filtered out, insert a
        period before the capital word.

        Args:
        reports (Dict[Tuple[str, str, str], str]): A dictionary containing the
            reports
        capital_words_filtered (List[str]): A list containing the words that
            are able to be filtered out

        Returns:
        Dict[Tuple[str, str, str], str]: A dictionary containing the reports
            with the capital words adjusted
        """
        counter = 0
        for key, report in reports.items():
            offset = -1
            matches = list(re.finditer(r"(?<!^)(?<![.!?]\s)([A-Z][\w&-]+)", report))

            for match in matches:
                word = match.group()
                if word not in capital_words_filtered:
                    if self.verbose:
                        print(f"Key: {key}")
                        print(f"Original report:  {report}")
                    insert_period_index = match.start() + offset
                    report = (
                        report[:insert_period_index]
                        + "."
                        + report[insert_period_index:]
                    )

                    offset += len(".")
                    counter += 1

                    if self.verbose:
                        print(f"Corrected report: {report}")
                        print("")

            reports[key] = report
        print(
            f"Inserted {counter} periods ('.') in the text based on capital "
            f"word analysis."
        )
        return reports

    def comma_capital_adjustment(self, reports, capital_words_filtered):
        """
        If you observe ", [A-Z][\\w&-]*" in the text, insert a period instead of
        the comma.

        Args:
        reports (Dict[Tuple[str, str, str], str]): A dictionary containing the
            reports
        capital_words_filtered (List[str]): A list containing the words that
            are able to be filtered out

        Returns:
        Dict[Tuple[str, str, str], str]: A dictionary containing the reports
            with the commas adjusted to periods
        """
        comma_pattern = r", ([A-Z][\w&-]*)"
        counter = 0
        for key, report in reports.items():
            matches = list(re.finditer(comma_pattern, report))
            for match in matches:
                word = match.group(1)
                if word not in capital_words_filtered:
                    if self.verbose:
                        print(f"Key: {key}")
                        print(f"Original report:  {report}")

                    insert_period_index = match.start(1)
                    report = (
                        report[:insert_period_index]
                        + "."
                        + report[insert_period_index + 1:]
                    )
                    counter += 1

                    if self.verbose:
                        print(f"Corrected report: {report}")
                        print("")

            reports[key] = report
        print(f"Amount of commas adjusted to periods: {counter}")
        return reports

    def remove_given_sentence_heuristic(self, reports):
        """
        Heuristic to remove the sentences that start with 'Given ... , [A-Z]'
        and 'Given ... ,$'. This is done to remove sentences that are not
        relevant for the reports.

        Args:
        preprocessed_reports_per_section (Dict[Tuple[str, str, str], str]):
            A dictionary containing the preprocessed reports

        Returns:
        Dict[Tuple[str, str, str], str]: A dictionary containing the
            preprocessed reports with the sentences removed
        """
        counter = 0

        given_pattern_1 = r"^Given.*, [A-Z]"

        given_pattern_2 = r"^Given.*,$"

        given_pattern_3 = r"^Given.*\,\.$"

        given_pattern = f"({given_pattern_1})|({given_pattern_2})|({given_pattern_3})"

        for key, report in reports.items():
            sentences = re.split(r"(?<=[.!?])\s+", report)
            new_sentences = []
            for sentence in sentences:
                if not re.match(given_pattern, sentence.strip()):
                    new_sentences.append(sentence)
                else:
                    if self.verbose:
                        print(f"Key: {key}")
                        print(f"Original report:  {report}")
                        print(f"Removed sentence: {sentence}")
                        print("")
                    counter += 1
            reports[key] = ". ".join(new_sentences)

        print(f"Amount of sentences removed that start with 'Given': " f"{counter}")
        return reports


class TextPreprocessor:
    def __init__(
        self,
        medical_reports_per_patient,
        num_servers=1,
        verbose=False,
        use_heuristics=True,
    ):
        self.medical_reports_per_patient = medical_reports_per_patient
        self.num_servers = num_servers
        self.verbose = verbose
        self.use_heuristics = use_heuristics

        # Initialize helper classes
        self.data_handler = DataHandler(
            self.medical_reports_per_patient, self.use_heuristics, self.verbose
        )
        self.report_analyzer = ReportAnalyzer(self.medical_reports_per_patient)
        self.heuristcs_processor = HeuristicsProcessor(self.verbose)

        # Define the rules that are considered good for automatic correction
        self.is_good_rule = lambda rule: (
            (rule.ruleId == "UPPERCASE_SENTENCE_START" and len(rule.replacements))
            or (rule.category == "TYPOGRAPHY" and len(rule.replacements))
            or (
                rule.category == "PUNCTUATION"
                and (
                    rule.ruleId == "DOUBLE_PUNCTUATION"
                    or rule.ruleId == "EN_UNPAIRED_BRACKETS"
                    or rule.ruleId == "UNLIKELY_OPENING_PUNCTUATION"
                )
                and len(rule.replacements)
            )
            or (
                rule.category == "GRAMMAR"
                and rule.ruleId == "PHRASE_REPETITION"
                and len(rule.replacements)
            )
            or (rule.category == "MISC" and rule.ruleId == "ENGLISH_WORD_REPEAT_RULE")
        )

        self.grammar_checker = GrammarChecker(
            num_servers=self.num_servers,
            is_good_rule=self.is_good_rule,
            verbose=self.verbose,
        )

        self.capital_words_filtered = []

    def main_preprocessing(self):
        # Flatten the dictionary to the level (patient, report_nr, section)
        flat_reports = self.data_handler.flatten_dict(
            self.medical_reports_per_patient, level=3
        )

        # Join the sentences of each section together
        preprocessed_reports = self.data_handler.preprocess_each_section(flat_reports)

        # Check the grammar of the reports
        corrected_reports = self.grammar_checker.grammar_check_report(
            preprocessed_reports
        )

        # Change HE to H&E
        corrected_reports = {
            tuple: re.sub(r"\bHE\b", "H&E", content)
            for tuple, content in corrected_reports.items()
        }

        if self.use_heuristics:
            # Capital word analysis
            self.capital_words_filtered = self.report_analyzer.capital_word_analysis(
                corrected_reports
            )

            # TODO Remove
            print("These words are able to be in the middle of a sentence:")
            print(self.capital_words_filtered)

            # Heuristic 1: Capital word adjustment
            corrected_reports = self.heuristcs_processor.capital_word_adjustment(
                corrected_reports, self.capital_words_filtered
            )

            # Heuristic 2: Remove sentence of specific structure that contain
            # the word 'given'
            corrected_reports = (
                self.heuristcs_processor.remove_given_sentence_heuristic(
                    corrected_reports
                )
            )

            # Heuristic 3: Comma to period adjustment
            corrected_reports = self.heuristcs_processor.comma_capital_adjustment(
                corrected_reports, self.capital_words_filtered
            )

        # Unflatten the dictionary
        unflattened_reports = self.data_handler.unflatten_dict(
            corrected_reports, level=3
        )

        # Join the sections
        joined_reports = self.data_handler.join_sections(unflattened_reports)

        # Flatten the dictionary to the level (patient, report_nr)
        flat_joined_reports = self.data_handler.flatten_dict(joined_reports, level=2)

        # Remove duplicate sentences
        duplicates = 0
        for key, content in flat_joined_reports.items():
            flat_joined_reports[key], inner_duplicates = (
                self.data_handler.delete_duplicate_sentences(content)
            )
            duplicates += inner_duplicates

        print(f"Removed {duplicates} duplicate sentences.")

        # Final grammar check
        final_corrected_reports = self.grammar_checker.grammar_check_report(
            flat_joined_reports
        )

        # Some finishing touches
        if self.use_heuristics:
            # Replace only the exact `,.` sequence with `.`
            final_corrected_reports = {
                tuple: re.sub(r",\.", ".", content)
                for tuple, content in final_corrected_reports.items()
            }

            # Replace only the exact `;.` sequence with `.`
            final_corrected_reports = {
                tuple: re.sub(r";\.", ".", content)
                for tuple, content in final_corrected_reports.items()
            }

            # Replace ' : ' with ': '
            final_corrected_reports = {
                tuple: re.sub(r" : ", ": ", content)
                for tuple, content in final_corrected_reports.items()
            }

        # Unflatten the dictionary
        final_unflattened_reports = self.data_handler.unflatten_dict(
            final_corrected_reports, level=2
        )

        return final_unflattened_reports

    def print_random_reports(self, reports, n):
        """
        Print n random reports. This is done to check if the preprocessing has
        been done correctly.

        Args:
        n (int): The number of reports that should be printed

        Returns:
        None
        """
        for _ in range(n):
            random_choice = random.choice(list(reports.keys()))
            wrapped_string = textwrap.fill(str(reports[random_choice]), width=79)
            print(wrapped_string)
            print("\n\n")
