## Evaluation Metrics Implementation Summary

### Changes Made to IR_Project.py

#### 1. Enhanced `evaluate_retrieval()` Method
- **Location**: Updated the existing method in the `WikipediaIRSystem` class
- **New Metrics Implemented**:
  - **Precision**: Calculates the ratio of relevant documents retrieved to total documents retrieved
    - Formula: Precision = True Positives / Retrieved Documents
  - **Recall**: Calculates the ratio of relevant documents retrieved to total relevant documents
    - Formula: Recall = True Positives / Relevant Documents
  - **Mean Average Precision (MAP)**: Measures the average precision across different recall levels
    - Formula: MAP = Sum of (Precision at each relevant position) / Total Relevant Documents

#### 2. Added Menu Option
- Added option **"7. Evaluate Retrieval Performance"** to the query interface
- Updated the exit option from 7 to 8

#### 3. Interactive Evaluation Interface
Users can now:
- Enter a search query
- Specify relevant document titles (comma-separated)
- View detailed evaluation metrics with clear formatting
- See breakdown of true positives, precision, recall, and MAP scores

### Metrics Explanation

**Precision (0.0 - 1.0)**
- Measures accuracy of retrieved results
- What percentage of retrieved documents are actually relevant?
- High precision = few false positives
- Example: 0.4000 means 40% of retrieved documents are relevant

**Recall (0.0 - 1.0)**
- Measures completeness of retrieval
- What percentage of relevant documents did we find?
- High recall = few false negatives
- Example: 0.6667 means we found 66.67% of the relevant documents

**MAP - Mean Average Precision (0.0 - 1.0)**
- Considers both precision and the ranking of relevant documents
- Higher values when relevant documents appear earlier in results
- Accounts for multiple relevant documents at different positions
- Example: 0.4667 indicates moderate ranking quality

### Test Results (test_evaluation.py)

Three test cases were executed demonstrating the metrics:

**Test Case 1: Machine Learning & AI Query**
- Precision: 0.4000 (2 out of 5 retrieved are relevant)
- Recall: 0.6667 (2 out of 3 relevant documents found)
- MAP: 0.4667 (moderate ranking)

**Test Case 2: Deep Learning & Neural Networks Query**
- Precision: 0.6000 (3 out of 5 retrieved are relevant)
- Recall: 1.0000 (all 3 relevant documents found)
- MAP: 1.0000 (perfect ranking - all relevant docs at top)

**Test Case 3: Data Science & Analytics Query**
- Precision: 0.6000 (3 out of 5 retrieved are relevant)
- Recall: 1.0000 (all 3 relevant documents found)
- MAP: 0.8667 (very good ranking)

### Usage in IR_Project.py

When running the main program:
1. Select option "7. Evaluate Retrieval Performance"
2. Enter your search query
3. Enter relevant document titles (from your collected Wikipedia articles)
4. View detailed metrics with visual formatting

### Files Modified
- `IR_Project.py` - Enhanced evaluation method and added menu option
- `test_evaluation.py` - New test script demonstrating all three metrics

### Output Format Example
```
============================================================
EVALUATION METRICS
============================================================

Query: 'machine learning and artificial intelligence'
Number of relevant documents: 3
Number of documents retrieved: 5
Relevant documents retrieved: 2

────────────────────────────────────────────────────────────
EVALUATION METRICS:
────────────────────────────────────────────────────────────
Precision    : 0.4000 (2/5)
Recall       : 0.6667 (2/3)
MAP (Mean Average Precision): 0.4667
────────────────────────────────────────────────────────────
```
