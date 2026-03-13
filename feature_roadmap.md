# Feature Roadmap: Pull Request Documentation Generator

Yes! It is **absolutely possible** to generate cleanly formatted Word documents from Pull Requests (PRs) using our existing Portable Python setup. Your `qwen2.5:14b` model is exceptionally well-suited for summarizing code changes into human-readable business documentation.

Here is a detailed breakdown answering your questions and outlining the roadmap for this feature.

---

## Technical Feasibility & Your Questions

### 1. Does the Local LLM have the ability to go through external links?

**No, not natively.** A local LLM is just a brain—it cannot "browse the web" or click a GitHub/GitLab link by itself.
*However*, we can write Python code in our CLI that **does** fetch the data from the link (using GitHub/GitLab APIs or web scraping) and then feeds that text directly into the LLM as context.

### 2. How will we send the input? (Links vs. Screenshots)

**Sending the PR Link (or PR Number) is the best and most robust approach.**
If your code is hosted on a platform like GitHub, GitLab, or Bitbucket, our Python script can use their REST API to download the **PR Diff** (the exact lines of code added/removed) and the **PR Title/Description**.

*Why avoid screenshots?*

1. **Context Limits:** A screenshot of a 50-file PR is impossible. A text-diff handles thousands of lines easily.
2. **Speed & Accuracy:** Vision models are slower, heavier, and prone to "hallucinating" text from blurry images. The raw text of a Git Diff ensures 100% accuracy of what actually changed.
   *If you still want a Vision Model eventually, the best open-source choice right now is **Llama 3.2 Vision (11B)**.*

### 3. How do we generate the Word document from the model?

The LLM will generate the raw text summary in a structured format (like Markdown). We will then use the open-source Python library **`python-docx`** to programmatically take that text, apply styles (Headers, Bullet points, bold text), and save it directly to your SSD as a `.docx` file!

*(Note: We actually already installed `python-docx` into your Mac virtual environment during our baseline setup!)*

---

## The Roadmap

We can build this as a dedicated "Mode" inside our existing `cli.py` or as a new standalone script (e.g., `generate_pr_doc.py`). Here is the step-by-step plan:

### Step 1: Secure API Access

To read your PRs programmatically, the Python script needs access to your repository.

- **Action:** You will generate a Personal Access Token (PAT) for your git provider (GitHub, GitLab, etc.).
- **Action:** We will add a feature to the CLI to securely accept and store this token on your SSD.

### Step 2: The Fetcher Module

We will write a Python function that accepts a PR URL (or ID) and automatically fetches:

- The PR Title and Description.
- The list of files changed.
- The raw `git diff` (the actual code additions and deletions).

### Step 3: The LLM Summarization Prompt

We will design a specialized prompt for `qwen2.5:14b`. We will feed it the fetched diff and ask it to output a structured summary.
**Example Prompt Structure:**

> "You are a Senior Technical Writer. Analyze the following code changes from this Pull Request and generate a release notes document. Include an Executive Summary, a Technical Breakdown by file, and any Potential Impacts. Do not output code blocks."

### Step 4: The Word Document Generator

We will write a function using `python-docx` that:

- Reads the LLM's response.
- Creates a new Word Document.
- Formats it beautifully with your company/project heading, timestamps, and structured sections.
- Saves it to `/Volumes/Sahith_SSD/IDE_Expert_Project/data/docs/`.

### Step 5: Integration into Nexus CLI

We will update `cli.py` to include a new menu option or terminal command.
Example Flow:

1. `Select Option: p` (Generate PR Doc)
2. `Enter PR URL:` `https://github.com/your-org/repo/pull/123`
3. *Nexus thinks for 30 seconds...*
4. *"Success! Your documentation has been saved to PR_123_Summary.docx"*

---

## Do you approve this approach?

If you like this roadmap, please let me know which Git Provider your team uses (e.g., GitHub, GitLab, Bitbucket, Azure DevOps). I can begin writing the fetcher and document generator modules immediately!
