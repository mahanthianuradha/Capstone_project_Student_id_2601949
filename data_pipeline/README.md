# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end data pipeline using the public scraping-practice website **Books to Scrape**.

The pipeline follows this workflow:

    Web Scraping
         ↓
    Data Cleaning
         ↓
    Data Type Conversion
         ↓
    GBP → INR Conversion
         ↓
    Normalized SQLite Database
         ↓
    SQL Queries
         ↓
    Pandas Analysis
         ↓
    SQL JOIN vs Pandas Merge Validation

The objective is to demonstrate a complete catalog-style data-engineering workflow: scrape raw product data, clean and transform it, enrich it using the project's fixed currency conversion rate, store it in a normalized relational database, and query the resulting data using SQL and pandas.

---

## 1. Data Source

The data source used for this project is:

**Books to Scrape**

https://books.toscrape.com/

Books to Scrape is a public website specifically designed for practicing web scraping.

The website does not require:

- Login
- API key
- Paid subscription

The scraping implementation uses:

- `requests` to retrieve HTML pages
- `BeautifulSoup` to parse HTML content

---

## 2. Project Structure

The Module 1 folder is organized as follows:

    data_pipeline/
    │
    ├── pipeline.py
    ├── books.db
    ├── README.md
    ├── requirements.txt
    │
    └── outputs/
        ├── cleaned_books.csv
        ├── query_01_output.csv
        ├── query_02_output.csv
        ├── query_03_output.csv
        ├── query_04_output.csv
        ├── query_05_output.csv
        └── join_comparison.csv

The SQLite database can either be committed to the repository or regenerated from scratch by running `pipeline.py`.

---

## 3. Installation Requirements

Python 3.9 or later is recommended.

Install the required packages using:

    pip install -r requirements.txt

The main Python packages used by this module are:

    requests
    beautifulsoup4
    pandas
    numpy

SQLite is provided through Python's built-in `sqlite3` module, so a separate SQLite installation is not required.

---

## 4. Running the Pipeline

From the project root, run:

    python data_pipeline/pipeline.py

Alternatively:

    cd data_pipeline
    python pipeline.py

The pipeline executes the complete workflow automatically:

1. Scrape books from Books to Scrape.
2. Collect the required raw fields.
3. Clean and convert the scraped fields.
4. Convert GBP prices to INR.
5. Create the SQLite database.
6. Create the normalized database tables.
7. Insert categories and books.
8. Execute the required SQL queries.
9. Save query outputs.
10. Read SQL results into pandas using `pd.read_sql()`.
11. Reproduce the SQL JOIN using `pd.merge()`.
12. Compare the SQL JOIN result with the pandas merge result.
13. Perform validation checks.

No manual copy-pasting of scraped data is required.

---

## 5. Scraping Scope

The pipeline scrapes books across multiple categories.

For every book, the following raw fields are collected:

| Field          | Description                                        |
| -------------- | -------------------------------------------------- |
| `title`        | Book title                                         |
| `price`        | Price as displayed by the website in GBP           |
| `star_rating`  | Rating text such as One, Two, Three, Four, or Five |
| `availability` | Availability text displayed by the website         |
| `category`     | Book category                                      |

The final dataset contains at least:

- 60 books
- 3 different categories

This satisfies the required scraping scope for Module 1.

---

## 6. Raw Data Collection

The pipeline uses `requests` to retrieve the web pages.

Example approach:

    response = requests.get(url)
    response.raise_for_status()

`BeautifulSoup` is then used to parse the returned HTML:

    soup = BeautifulSoup(response.text, "html.parser")

The parser extracts the required book information from the product listing pages.

HTTP errors and unexpected page structures are handled defensively so that a temporary malformed response does not silently produce incorrect data.

---

## 7. Data Cleaning

The raw scraped values require cleaning before they can be used for analysis or inserted into the relational database.

The following transformations are performed:

- Price string → `price_gbp` float
- Star-rating text → `rating` integer
- Availability text → `in_stock` Boolean
- Category → normalized category table
- GBP price → `price_inr`

---

## 8. Price Cleaning

The website displays prices with the GBP currency symbol.

For example:

    £51.77

The currency symbol is removed and the remaining value is converted to a floating-point number:

    £51.77
       ↓
    51.77

The cleaned column is:

    price_gbp

The expected data type is:

    float

The cleaning operation conceptually performs:

    price_gbp = float(price_text.replace("£", "").strip())

This makes the price suitable for numerical analysis and currency conversion.

---

## 9. Star Rating Conversion

The website provides ratings as text.

Examples include:

    One
    Two
    Three
    Four
    Five

These values are converted to integers:

| Original value | Converted value |
| -------------- | --------------: |
| One            |               1 |
| Two            |               2 |
| Three          |               3 |
| Four           |               4 |
| Five           |               5 |

The resulting column is:

    rating

The expected values are integers from 1 to 5.

The mapping used is:

    One   → 1
    Two   → 2
    Three → 3
    Four  → 4
    Five  → 5

---

## 10. Availability Conversion

The website provides availability as text.

For example:

    In stock

This is converted into a Boolean column:

    in_stock

For example:

    In stock
       ↓
    True

The final pandas column uses Boolean values.

When stored in SQLite, Boolean values are represented using integers:

    True  → 1
    False → 0

This is consistent with SQLite's Boolean representation.

Unexpected availability text is handled defensively instead of causing the entire pipeline to crash.

---

## 11. Handling Parsing Errors

Web scraping pipelines must be able to deal with unexpected values.

The pipeline therefore handles parsing errors defensively.

For numeric fields such as:

- `price_gbp`
- `rating`

invalid values are converted to missing values during parsing.

Where a numeric value cannot be parsed, the pipeline uses median imputation rather than crashing the complete pipeline.

The general approach is:

    Valid value
        ↓
    Use parsed value

    Invalid numeric value
        ↓
    Convert to missing
        ↓
    Median imputation

If a record is completely unusable or cannot be meaningfully recovered, the row may be dropped.

This approach ensures that one malformed web record does not prevent the entire dataset from being processed.

The pipeline reports any rows or values affected by the cleaning process.

---

## 12. Fixed Currency Conversion

The project requires a fixed baseline currency conversion rate.

The required rate is:

    1 GBP = 105.50 INR

This is an artificial, project-defined constant for this assignment.

It is NOT:

- a live exchange rate
- a historical exchange rate
- a market rate

Therefore, no external currency API is required.

The conversion is calculated as:

    price_inr = price_gbp * 105.50

For example:

    £10.00 × 105.50 = ₹1,055.00

The resulting column is:

    price_inr

The fixed rate is deliberately used instead of a live API because the project specifically requires the fixed-rate baseline.

This makes the pipeline deterministic and reproducible.

---

## 13. Currency Conversion Validation

The pipeline validates that `price_inr` has been calculated from the required fixed rate.

Conceptually:

    expected_price_inr = price_gbp * 105.50

The resulting values are compared with the stored `price_inr` values.

A numerical tolerance such as `numpy.isclose()` may be used because floating-point calculations can introduce very small representation differences.

Example:

    np.allclose(
        df["price_inr"],
        df["price_gbp"] * 105.50
    )

This confirms that the required project conversion rate has been applied.

---

## 14. Database Design

SQLite is used as the relational database.

The database file is:

    books.db

The database uses a normalized two-table structure:

    categories
         │
         │ category_id
         ↓
       books

The design separates category information from individual book records.

---

## 15. Categories Table

The categories table stores each unique category once.

Schema:

    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT UNIQUE NOT NULL
    );

Columns:

| Column          | Type    | Description          |
| --------------- | ------- | -------------------- |
| `category_id`   | INTEGER | Primary key          |
| `category_name` | TEXT    | Unique category name |

The `UNIQUE` constraint prevents duplicate category names.

---

## 16. Books Table

The books table stores the cleaned book information.

Schema:

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER,
        FOREIGN KEY (category_id)
            REFERENCES categories(category_id)
    );

Columns:

| Column        | Type    | Description                      |
| ------------- | ------- | -------------------------------- |
| `book_id`     | INTEGER | Primary key                      |
| `title`       | TEXT    | Book title                       |
| `price_gbp`   | REAL    | Cleaned GBP price                |
| `price_inr`   | REAL    | Converted INR price              |
| `rating`      | INTEGER | Rating from 1 to 5               |
| `in_stock`    | INTEGER | SQLite representation of Boolean |
| `category_id` | INTEGER | Foreign key to `categories`      |

---

## 17. Primary Key / Foreign Key Relationship

The two tables are connected using:

    categories.category_id
             │
             │ 1-to-many
             ↓
    books.category_id

`categories.category_id` is the primary key.

`books.category_id` is the foreign key.

This means that one category can have many books.

For example:

    Travel
       ├── Book A
       ├── Book B
       └── Book C

The category name is stored once in the `categories` table instead of being duplicated for every book.

This provides a normalized relational design.

---

## 18. Database Loading

The database is created using Python's built-in `sqlite3` module.

The loading process performs the following operations:

1. Connect to SQLite.
2. Create the `categories` table.
3. Create the `books` table.
4. Extract unique categories from the cleaned DataFrame.
5. Insert categories into the categories table.
6. Retrieve category IDs.
7. Map category names to category IDs.
8. Insert cleaned books into the books table.
9. Commit the transaction.
10. Close the database connection.

The resulting database is therefore reproducible from the scraping script.

---

## 19. Final Cleaned Dataset

The final in-memory dataset contains fields such as:

| Column      | Type    | Description                      |
| ----------- | ------- | -------------------------------- |
| `title`     | string  | Book title                       |
| `price_gbp` | float   | Cleaned GBP price                |
| `price_inr` | float   | GBP price converted using 105.50 |
| `rating`    | integer | Rating from 1 to 5               |
| `in_stock`  | boolean | Stock availability               |
| `category`  | string  | Book category                    |

The database version separates category information into the normalized `categories` table.

---

## 20. SQL Queries

At least five SQL queries are executed against the SQLite database.

Together, the queries demonstrate:

- `SELECT`
- `WHERE`
- `ORDER BY`
- `LIMIT`
- `DISTINCT`
- `BETWEEN` or `IN`
- `JOIN`

The SQL query strings and their outputs are printed and saved.

---

## 21. Query 1 — SELECT and WHERE

Example:

    SELECT
        title,
        price_gbp,
        rating,
        in_stock
    FROM books
    WHERE in_stock = 1;

Purpose:

This query selects books that are currently in stock.

It demonstrates:

    SELECT
    WHERE

---

## 22. Query 2 — ORDER BY

Example:

    SELECT
        title,
        price_gbp,
        rating
    FROM books
    ORDER BY price_gbp DESC;

Purpose:

This query sorts books from the highest GBP price to the lowest.

It demonstrates:

    ORDER BY

---

## 23. Query 3 — LIMIT

Example:

    SELECT
        title,
        price_gbp,
        rating
    FROM books
    ORDER BY rating DESC, price_gbp DESC
    LIMIT 10;

Purpose:

This query returns the top 10 books according to rating and price ordering.

It demonstrates:

    LIMIT

---

## 24. Query 4 — DISTINCT

Example:

    SELECT DISTINCT
        category_name
    FROM categories
    ORDER BY category_name;

Purpose:

This query returns the unique categories in the database.

It demonstrates:

    DISTINCT

---

## 25. Query 5 — BETWEEN

Example:

    SELECT
        title,
        price_gbp,
        rating
    FROM books
    WHERE price_gbp BETWEEN 10 AND 30
    ORDER BY price_gbp;

Purpose:

This query returns books whose GBP price is between £10 and £30.

It demonstrates:

    BETWEEN

---

## 26. SQL JOIN Query

A JOIN is used to combine data from the normalized `books` and `categories` tables.

Example:

    SELECT
        b.book_id,
        b.title,
        b.rating,
        b.price_gbp,
        b.price_inr,
        b.in_stock,
        c.category_name
    FROM books AS b
    JOIN categories AS c
        ON b.category_id = c.category_id
    ORDER BY b.rating DESC, b.title
    LIMIT 10;

This query demonstrates the required relationship between the two database tables.

The result contains book information together with its category name.

---

## 27. SQL Query Outputs

The SQL query results are printed during pipeline execution.

They are also saved as CSV files for reproducibility and inspection.

Example output directory:

    outputs/
    ├── query_01_output.csv
    ├── query_02_output.csv
    ├── query_03_output.csv
    ├── query_04_output.csv
    ├── query_05_output.csv
    └── join_comparison.csv

These files provide evidence that the SQL queries were executed successfully.

---

## 28. Reading SQL Results with Pandas

At least two SQL query results are read into pandas using `pd.read_sql()`.

Example:

    result_df = pd.read_sql(
        "SELECT title, price_gbp, rating FROM books",
        connection
    )

This demonstrates that the SQLite database can be queried directly into a pandas DataFrame.

The SQL database therefore serves as the persistent relational layer while pandas can be used for subsequent analysis.

---

## 29. SQL JOIN vs Pandas Merge

The JOIN result is independently reproduced using pandas.

The relevant in-memory DataFrames are:

    books_df
    categories_df

The pandas equivalent is:

    merged_df = books_df.merge(
        categories_df,
        on="category_id",
        how="inner"
    )

The SQL JOIN and pandas merge are then compared.

The comparison verifies:

- Same number of rows
- Same columns
- Same values
- Same records

---

## 30. Preventing JOIN/Merge Ordering Issues

SQL and pandas do not necessarily return rows in exactly the same order unless ordering is explicitly enforced.

Therefore, before comparison, both results are normalized by:

1. Selecting the same columns.
2. Sorting using the stable `book_id`.
3. Resetting the DataFrame index.

Example:

    sql_result = sql_result[
        [
            "book_id",
            "title",
            "rating",
            "price_gbp",
            "price_inr",
            "in_stock",
            "category_name"
        ]
    ].sort_values(
        "book_id"
    ).reset_index(drop=True)

    merged_result = merged_result[
        [
            "book_id",
            "title",
            "rating",
            "price_gbp",
            "price_inr",
            "in_stock",
            "category_name"
        ]
    ].sort_values(
        "book_id"
    ).reset_index(drop=True)

The results can then be compared using:

    pd.testing.assert_frame_equal(
        sql_result,
        merged_result,
        check_dtype=False
    )

Using `check_dtype=False` avoids false mismatches caused only by differences between SQLite and pandas data-type representations.

The comparison verifies that the SQL JOIN and pandas merge produce equivalent data.

---

## 31. Data Validation

The pipeline includes validation checks to ensure the required acceptance criteria are met.

### Minimum number of books

    assert len(df) >= 60

### Minimum number of categories

    assert df["category"].nunique() >= 3

### Rating validation

    assert df["rating"].between(1, 5).all()

### Price validation

    assert df["price_gbp"].notna().all()

### INR validation

    assert df["price_inr"].notna().all()

### Currency conversion validation

    assert np.allclose(
        df["price_inr"],
        df["price_gbp"] * 105.50
    )

These checks help ensure that an incomplete or incorrectly transformed dataset is not silently accepted.

---

## 32. Reproducibility

The pipeline is designed to run from scratch without manual intervention.

Running:

    python data_pipeline/pipeline.py

will recreate the pipeline outputs.

The process can recreate:

    books.db

and the associated query output files.

The SQLite database can therefore either be committed to the repository or regenerated from the Python script.

---

## 33. Design Decisions

### Web Scraping

`requests` was selected to retrieve the HTML pages and `BeautifulSoup` was selected to parse the HTML.

### Cleaning

Raw text fields are converted into appropriate analytical types before database insertion.

### Invalid Numeric Values

Invalid numeric values are handled using median imputation rather than crashing the entire pipeline.

This preserves otherwise valid records and provides a robust approach to occasional malformed values.

### Invalid/Unrecoverable Rows

If a record cannot be meaningfully recovered, it can be dropped rather than inserting unreliable data into the database.

### Currency Conversion

The project-defined fixed conversion rate is:

    1 GBP = 105.50 INR

No external currency API is used because the fixed rate is the required grading baseline.

### Database

SQLite was selected because it is lightweight, serverless, reproducible, and available through Python's standard library.

### Normalization

Categories are stored separately from books to reduce duplication and demonstrate a proper primary-key/foreign-key relationship.

### SQL and Pandas

SQL is used for relational querying while pandas is used for DataFrame-based analysis.

### JOIN Validation

The SQL JOIN is independently reproduced using `pd.merge()` and the results are sorted and normalized before comparison to avoid false mismatches caused by row ordering or database-specific data types.

---

## 34. Acceptance Criteria Checklist

| Requirement                                    | Status   |
| ---------------------------------------------- | -------- |
| Scrape using `requests`                        | Complete |
| Parse using `BeautifulSoup`                    | Complete |
| At least 60 books                              | Complete |
| At least 3 categories                          | Complete |
| Capture title                                  | Complete |
| Capture price                                  | Complete |
| Capture star rating                            | Complete |
| Capture availability                           | Complete |
| Capture category                               | Complete |
| `price_gbp` as float                           | Complete |
| `rating` as integer                            | Complete |
| `in_stock` as Boolean                          | Complete |
| Parsing failures handled                       | Complete |
| Median imputation for numeric parsing failures | Complete |
| GBP → INR conversion                           | Complete |
| Fixed rate = 105.50 INR                        | Complete |
| Two normalized SQLite tables                   | Complete |
| Primary-key/foreign-key relationship           | Complete |
| At least 5 SQL queries                         | Complete |
| SELECT / WHERE                                 | Complete |
| ORDER BY                                       | Complete |
| LIMIT                                          | Complete |
| DISTINCT                                       | Complete |
| IN / BETWEEN                                   | Complete |
| SQL JOIN                                       | Complete |
| Query outputs saved                            | Complete |
| `pd.read_sql()` demonstrated                   | Complete |
| `pd.merge()` demonstrated                      | Complete |
| SQL JOIN vs pandas merge validation            | Complete |
| Database reproducible from script              | Complete |

---

## 35. Final Pipeline

The completed Module 1 pipeline can be summarized as:

    Books to Scrape
          │
          ▼
    requests
          │
          ▼
    BeautifulSoup
          │
          ▼
    Raw Book Data
          │
          ▼
    Data Cleaning
          │
          ├── price → price_gbp
          ├── rating text → rating
          └── availability → in_stock
          │
          ▼
    Fixed Currency Conversion
          │
          │ 1 GBP = 105.50 INR
          ▼
    price_inr
          │
          ▼
    Normalized SQLite Database
          │
          ├── categories
          │
          └── books
                 │
                 ▼
             SQL Queries
                 │
                 ├── SELECT / WHERE
                 ├── ORDER BY
                 ├── LIMIT
                 ├── DISTINCT
                 ├── BETWEEN
                 └── JOIN
                         │
                         ▼
                    pandas read_sql()
                         │
                         ▼
                    pandas merge()
                         │
                         ▼
                 Equivalence Check

---

## 36. Conclusion

Module 1 implements a complete raw-to-relational catalog data pipeline.

The solution demonstrates how public web data can be:

1. Scraped programmatically.
2. Cleaned and converted into appropriate data types.
3. Enriched using a deterministic project-defined currency rate.
4. Stored using a normalized relational schema.
5. Queried using SQL.
6. Loaded into pandas for further analysis.
7. Independently reproduced using pandas merge operations.
8. Validated for consistency and completeness.

The required fixed currency conversion used throughout the project is:

    1 GBP = 105.50 INR

The resulting implementation provides a reproducible foundation for catalog-style pricing and availability analysis.
