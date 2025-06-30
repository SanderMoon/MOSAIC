## text_preprocessor.py and text_inspection.ipynb
`text_inspection.ipynb` is usiang the `text_preprocessor.py` file and the various processors in this file to clean up the raw text data. The text data is based on the method described in the [Report Preparation](https://github.com/RTLucassen/report_preprocessing) study. 

Some steps the script takes to clean the text data are as follows:

1. **Section Preprocessing** (`structure_section_preprocessing`):
   - Joins sentences with newline characters into a single string by replacing newline characters (`\n`) with periods.
   - Removes extra spaces and replaces semicolons (`;`) with periods.

2. **Joining Sentences**:
   - Combines lists of sentences into single strings for each report section.

3. **Delete Duplicate Sentences** (`delete_duplicate_sentences`):
   - Removes repeated sentences by comparing them, deleting the latter occurrence of any duplicates.

4. **Grammar Check**:
   - Corrects grammatical mistakes using the `language_tool_python` library, with a custom function (`is_good_rule`) to filter which rules to apply.

5. **Capital Word Adjustment**:
   - Inserts a period before a capital word that appears in the middle of a sentence if it is not filtered out.

6. **Comma-Capital Adjustment**:
   - Replaces commas with periods in places where a capital letter follows a comma.

7. **Remove 'Given' Sentences**:
   - Removes sentences that start with "Given" followed by a specific pattern, based on a heuristic designed for this dataset.

8. **Replace HE with H&E**:
   - Replaces occurrences of "HE" with "H&E".

9. **Remove Duplicates Again**:
    - A second pass to remove any remaining duplicate sentences after the sections have been joined.


## create_subset_of_patients.py

Given **two** json files containing reports of the following format:
{
    "Pxxxxxx": {
        "Txx-xxxxx_I": "SOME TEXT."
    },
    "Pxxxxxx": {
        "Txx-xxxxx_I": "SOME TEXT"
    },
}

The `create_subset_of_patients.py` will take all patient IDs and case IDS of one set and selects those patients and cases in the other set. 
The main use-case is that during our study there was one text dataset based on two text labels, and another one based on three.
In order to investigate the additional benefit of the third label we should create two datasets with the same patients and cases, but one set contains the text from two labels, while the other contains additional text from the third label. 

The labels specifically are based on those used in the [Report Preparation](https://github.com/RTLucassen/report_preprocessing) study.