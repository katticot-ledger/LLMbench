# LLM Bench

LLM Bench is a benchmarking tool designed to compare the performance of various Language Models (LLMs). It allows you to evaluate and compare different LLMs on a specific task or dataset, providing insights into their speed and accuracy.LLM Bench is inspired by PrivateGPT

## Prerequisites

- Python 3.9 or higher
- PDM package manager
- Pretrained language models in the `models` directory

## Installation

1. Clone this repository to your local machine.

2. Rename `example.env` to `.env` and edit the variables appropriately.

3. Install the required dependencies using PDM. Run the following command in your terminal:
   ```
   pdm sync
   ```

## Usage

1. Ensure that the documents you want to benchmark are located in the `source_documents` directory.

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory.

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

2. Run the ingestion script to index the documents:
   - Execute the following command in your terminal:
     ```
     python ingest.py
     ```

3. Run the main application to benchmark the LLMs:
   - Execute the following command in your terminal:
     ```
     python main.py
     ```

4. The application will start and display a Streamlit user interface in your browser.

5. Select the desired LLM model type:
   - Use the sidebar radio button to choose a specific model type from the available options.

6. Select a model:
   - Use the dropdown selector in the sidebar to choose a specific model for the selected model type.

7. Enter a query:
   - Type your query in the provided text input field.

8. Press the "Enter" or "Return" key to execute the query and initiate the benchmarking process.

9. The application will display the question, answer, and relevant documents for the given query, along with the execution time for the selected LLM model.

10. Repeat the process with different queries and models to compare their performance.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to submit a pull request or open an issue in the repository.

## License

This project is licensed under the [MIT License](LICENSE).

---
