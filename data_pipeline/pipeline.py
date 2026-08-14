from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# Configuration
# ============================================================

# Base website from which the book information will be scraped.
BASE_URL = "https://books.toscrape.com"

# Fixed conversion rate required by the project.
# All GBP prices will be converted to INR using this rate.
GBP_TO_INR = 105.50

# Categories that we want to scrape.
# The dictionary stores:
#     key   -> category name
#     value -> URL of the category page
#
# Three categories are selected so that the final dataset
# contains more than the minimum required 60 rows.
CATEGORY_URLS = {
    "Fiction": f"{BASE_URL}/catalogue/category/books/fiction_10/index.html",
    "Mystery": f"{BASE_URL}/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": (
        f"{BASE_URL}/catalogue/category/books/historical-fiction_4/index.html"
    ),
}


# ============================================================
# Project directories and output files
# ============================================================

# Path(__file__) gives the location of this Python file.
# resolve() converts it into an absolute path.
# parent gives the folder containing this Python file.
PROJECT_ROOT = Path(__file__).resolve().parent

# Create separate folders for input/data, database files,
# and final output files.
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = PROJECT_ROOT / "database"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# File where the cleaned dataset will be saved as CSV.
CSV_PATH = DATA_DIR / "cleaned_books.csv"

# SQLite database file.
DB_PATH = DATABASE_DIR / "books_catalog.db"


# Text files used to store the results of SQL queries
# and the pandas JOIN comparison.
SQL_OUTPUT_PATH = OUTPUT_DIR / "sql_outputs.txt"
PANDAS_OUTPUT_PATH = OUTPUT_DIR / "pandas_read_sql_outputs.txt"
MERGE_OUTPUT_PATH = OUTPUT_DIR / "merge_comparison.txt"


# Maximum number of seconds to wait for a web request.
# This prevents the program from waiting indefinitely.
REQUEST_TIMEOUT = 30


# The website represents ratings using words such as
# "One", "Two", "Three", etc.
#
# This dictionary converts those words into numbers
# that can be used for calculations and analysis.
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ============================================================
# Utility functions
# ============================================================

def ensure_directories() -> None:
    """Create required project directories if they do not exist."""

    # mkdir() creates the directory.
    # parents=True also creates missing parent directories.
    # exist_ok=True prevents an error if the directory already exists.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_soup(
    session: requests.Session,
    url: str,
) -> BeautifulSoup:
    """
    Download a web page and convert its HTML into BeautifulSoup.

    BeautifulSoup allows us to search the HTML using CSS selectors
    and extract the information we need.
    """

    # Send a GET request to download the web page.
    # A timeout is used so that the program does not wait forever.
    response = session.get(url, timeout=REQUEST_TIMEOUT)

    # If the server returns an HTTP error such as 404 or 500,
    # raise an exception instead of continuing with invalid data.
    response.raise_for_status()

    # Convert the downloaded HTML text into a BeautifulSoup object.
    return BeautifulSoup(response.text, "html.parser")


def parse_price(price_text: str) -> float | None:
    """
    Convert a price such as '£51.77' into a numeric value 51.77.

    Returning None means that the price could not be extracted.
    """

    # If the input is empty, there is nothing to parse.
    if not price_text:
        return None

    # Search for a number that may contain decimal places.
    # Example:
    #     £51.77 -> 51.77
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", price_text)

    # If no number was found, return None.
    if not match:
        return None

    try:
        # Convert the extracted text into a floating-point number.
        return float(match.group(1))
    except ValueError:
        # Return None if the conversion unexpectedly fails.
        return None


def parse_rating(rating_text: str) -> int | None:
    """
    Convert a textual rating such as 'Three' into integer 3.
    """

    # Check whether a rating value was provided.
    if not rating_text:
        return None

    # Look up the rating in the RATING_MAP dictionary.
    # If the rating is not found, return None.
    return RATING_MAP.get(rating_text.strip())


def parse_availability(availability_text: str) -> bool | None:
    """
    Convert availability text into a Boolean value.

    Examples:
        'In stock' -> True
        'In stock (5 available)' -> True
        'Out of stock' -> False
    """

    # Empty availability information cannot be interpreted.
    if not availability_text:
        return None

    # Convert the text to lowercase and remove unnecessary spaces.
    # This makes the comparison more reliable.
    normalized = " ".join(availability_text.lower().split())

    # Any text containing "in stock" is treated as available.
    if "in stock" in normalized:
        return True

    # Any text containing "out of stock" is treated as unavailable.
    if "out of stock" in normalized:
        return False

    # Return None when the website contains an unexpected format.
    return None


# ============================================================
# Web Scraping
# ============================================================

def scrape_category(
    session: requests.Session,
    category_name: str,
    start_url: str,
) -> list[dict[str, Any]]:
    """
    Scrape all books belonging to one category.

    The function also follows the category's pagination links
    until there are no more pages.

    Each book record contains:
        - title
        - price
        - star rating
        - availability
        - category
    """

    # This list will store all books found in this category.
    rows: list[dict[str, Any]] = []

    # Start scraping from the first page of the category.
    current_url = start_url

    # Continue looping while there is another page to scrape.
    while current_url:
        print(f"Scraping: {current_url}")

        # Download the current page and parse its HTML.
        soup = get_soup(session, current_url)

        # Each book on the website is represented by
        # an article element with the CSS class "product_pod".
        products = soup.select("article.product_pod")

        # If no products are found, something unexpected happened.
        # Stop rather than silently creating an incomplete dataset.
        if not products:
            raise RuntimeError(
                f"No products found on category page: {current_url}"
            )

        # Process each book found on the current page.
        for product in products:

            # Find the HTML elements containing the required fields.
            title_element = product.select_one("h3 a")
            price_element = product.select_one("p.price_color")
            rating_element = product.select_one("p.star-rating")
            availability_element = product.select_one("p.availability")

            # The title is stored inside the "title" HTML attribute.
            # If the element does not exist, use an empty string.
            title = (
                title_element.get("title", "").strip()
                if title_element
                else ""
            )

            # Extract the displayed price, for example "£51.77".
            price = (
                price_element.get_text(" ", strip=True)
                if price_element
                else ""
            )

            # Get the CSS classes associated with the rating element.
            #
            # Example:
            # ["star-rating", "Three"]
            #
            # The second class contains the actual rating.
            rating = (
                " ".join(rating_element.get("class", []))
                if rating_element
                else ""
            )

            # Extract only the rating word, such as "Three".
            #
            # The first CSS class is "star-rating".
            # The second CSS class is the actual rating.
            rating_text = (
                rating_element.get("class", ["", ""])[1]
                if rating_element
                and len(rating_element.get("class", [])) > 1
                else ""
            )

            # Extract availability information such as
            # "In stock" or "Out of stock".
            availability = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else ""
            )

            # Store the extracted information as one dictionary.
            # Each dictionary represents one book.
            rows.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": rating_text,
                    "availability": availability,
                    "category": category_name,
                }
            )

        # Look for the "next" pagination button.
        next_link = soup.select_one("li.next a")

        # If a next page exists, create the complete URL.
        if next_link and next_link.get("href"):
            current_url = requests.compat.urljoin(
                current_url,
                next_link["href"],
            )
        else:
            # No next page means that scraping for this category is complete.
            current_url = None

    # Return all books collected from the category.
    return rows


def scrape_all_categories() -> pd.DataFrame:
    """Scrape all configured categories and combine them into one DataFrame."""

    # A Session allows multiple HTTP requests to reuse the same connection
    # and is generally more efficient than making completely separate requests.
    session = requests.Session()

    # This list will contain the rows from all categories.
    all_rows: list[dict[str, Any]] = []

    # Loop through every category defined in CATEGORY_URLS.
    for category_name, category_url in CATEGORY_URLS.items():

        # Scrape all pages belonging to the current category.
        category_rows = scrape_category(
            session=session,
            category_name=category_name,
            start_url=category_url,
        )

        print(
            f"Collected {len(category_rows)} rows "
            f"from {category_name}"
        )

        # Add the category's rows to the overall dataset.
        all_rows.extend(category_rows)

    # Convert the list of dictionaries into a pandas DataFrame.
    return pd.DataFrame(all_rows)


# ============================================================
# Data Cleaning and Transformation
# ============================================================

def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the scraped data and create analysis-ready columns.

    Main transformations:
        1. Clean text values.
        2. Convert price from text to GBP number.
        3. Convert GBP price to INR.
        4. Convert rating words to integers.
        5. Convert availability to Boolean.
        6. Validate the final dataset.
    """

    # Make a copy so that the original scraped DataFrame
    # is not changed directly.
    df = raw_df.copy()

    # --------------------------------------------------------
    # Basic text cleanup
    # --------------------------------------------------------

    # These columns contain text scraped from the website.
    text_columns = [
        "title",
        "price",
        "star_rating",
        "availability",
        "category",
    ]

    # Convert each text column to pandas string type
    # and remove leading/trailing spaces.
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # --------------------------------------------------------
    # Price conversion
    # --------------------------------------------------------

    # Convert strings such as "£51.77" into numeric values.
    # Invalid values become None.
    df["price_gbp"] = df["price"].apply(parse_price)

    # Calculate the median valid price.
    # The median will be used if a price is missing or invalid.
    price_median = df["price_gbp"].median()

    # Replace missing price values with the median price.
    df["price_gbp"] = df["price_gbp"].fillna(price_median)

    # --------------------------------------------------------
    # Rating conversion
    # --------------------------------------------------------

    # Convert rating words such as "Four" into the number 4.
    df["rating"] = df["star_rating"].apply(parse_rating)

    # Find the median rating for possible missing values.
    rating_median = df["rating"].median()

    # Fill missing ratings with the median.
    # round() makes sure the result is a whole number.
    # astype("int64") converts the result into integer type.
    df["rating"] = (
        df["rating"]
        .fillna(rating_median)
        .round()
        .astype("int64")
    )

    # Safety check:
    # Ratings should always be between 1 and 5.
    df["rating"] = df["rating"].clip(lower=1, upper=5)

    # --------------------------------------------------------
    # Availability conversion
    # --------------------------------------------------------

    # Convert availability text into True/False.
    df["in_stock"] = df["availability"].apply(parse_availability)

    # Count rows where availability could not be understood.
    invalid_availability = df["in_stock"].isna().sum()

    if invalid_availability:
        print(
            f"Dropping {invalid_availability} rows with "
            "unparseable availability."
        )

        # Availability is a required business field.
        # If it cannot be interpreted, remove that row.
        df = df.dropna(subset=["in_stock"]).copy()

    # Convert the remaining values into Boolean type.
    df["in_stock"] = df["in_stock"].astype(bool)

    # --------------------------------------------------------
    # GBP to INR conversion
    # --------------------------------------------------------

    # Multiply every GBP price by the fixed conversion rate.
    # round(2) keeps the INR value to two decimal places.
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # --------------------------------------------------------
    # Final column ordering
    # --------------------------------------------------------

    # Arrange the columns in a clear order for the final CSV.
    df = df[
        [
            "title",
            "price",
            "price_gbp",
            "price_inr",
            "star_rating",
            "rating",
            "availability",
            "in_stock",
            "category",
        ]
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    # The assignment requires at least 60 cleaned records.
    if len(df) < 60:
        raise RuntimeError(
            f"Only {len(df)} cleaned rows remain. "
            "At least 60 are required."
        )

    # Make sure all three requested categories are present.
    if df["category"].nunique() < 3:
        raise RuntimeError(
            "Fewer than three categories remain."
        )

    # Confirm that price_gbp is stored as a floating-point column.
    if not pd.api.types.is_float_dtype(df["price_gbp"]):
        raise TypeError("price_gbp must be float.")

    # Confirm that rating is stored as an integer.
    if not pd.api.types.is_integer_dtype(df["rating"]):
        raise TypeError("rating must be integer.")

    # Confirm that availability is stored as Boolean.
    if not pd.api.types.is_bool_dtype(df["in_stock"]):
        raise TypeError("in_stock must be boolean.")

    # Confirm that the INR price is stored as a floating-point value.
    if not pd.api.types.is_float_dtype(df["price_inr"]):
        raise TypeError("price_inr must be float.")

    # Return the cleaned and validated DataFrame.
    return df


# ============================================================
# SQLite Database
# ============================================================

def create_database() -> None:
    """Create the normalized two-table SQLite database."""

    # If an old database already exists, delete it.
    # This ensures every run starts with a fresh database.
    if DB_PATH.exists():
        DB_PATH.unlink()

    # Open a connection to the SQLite database.
    # The "with" statement automatically closes the connection.
    with sqlite3.connect(DB_PATH) as connection:

        # Enable foreign-key checking in SQLite.
        connection.execute("PRAGMA foreign_keys = ON")

        # Create the two required tables.
        connection.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            );
            """
        )


def load_database(df: pd.DataFrame) -> None:
    """
    Load the cleaned DataFrame into the normalized SQLite database.

    Categories are stored separately from books.
    Each book stores category_id instead of repeating the category name.
    """

    # Open the SQLite database connection.
    with sqlite3.connect(DB_PATH) as connection:

        # Enable foreign-key validation.
        connection.execute("PRAGMA foreign_keys = ON")

        # ----------------------------------------------------
        # Create the category table data
        # ----------------------------------------------------

        # Select only the category column.
        # drop_duplicates() ensures each category is inserted once.
        categories_df = (
            df[["category"]]
            .drop_duplicates()
            .rename(columns={"category": "category_name"})
            .sort_values("category_name")
            .reset_index(drop=True)
        )

        # Insert the categories into the SQLite table.
        categories_df.to_sql(
            "categories",
            connection,
            if_exists="append",
            index=False,
        )

        # ----------------------------------------------------
        # Create a category lookup
        # ----------------------------------------------------

        # Read category IDs generated by SQLite.
        # We need these IDs when inserting books.
        category_lookup = pd.read_sql(
            """
            SELECT category_id, category_name
            FROM categories
            """,
            connection,
        )

        # ----------------------------------------------------
        # Connect books to their category IDs
        # ----------------------------------------------------

        # Merge the original book data with the category IDs.
        #
        # many_to_one means:
        #     many books can belong to one category.
        books_df = df.merge(
            category_lookup,
            left_on="category",
            right_on="category_name",
            how="inner",
            validate="many_to_one",
        )

        # Keep only columns required by the books table.
        books_df = books_df[
            [
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category_id",
            ]
        ].copy()

        # SQLite does not have a separate Boolean storage type.
        # Boolean values are stored as 0 and 1.
        books_df["in_stock"] = books_df["in_stock"].astype(int)

        # Insert the books into the books table.
        books_df.to_sql(
            "books",
            connection,
            if_exists="append",
            index=False,
        )

        # ----------------------------------------------------
        # Database integrity checks
        # ----------------------------------------------------

        # Count the number of categories inserted.
        category_count = connection.execute(
            "SELECT COUNT(*) FROM categories"
        ).fetchone()[0]

        # Count the number of books inserted.
        book_count = connection.execute(
            "SELECT COUNT(*) FROM books"
        ).fetchone()[0]

        # Find books that do not have a matching category.
        # Such records would be called orphaned records.
        orphan_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM books b
            LEFT JOIN categories c
                ON b.category_id = c.category_id
            WHERE c.category_id IS NULL
            """
        ).fetchone()[0]

        print(f"Database categories: {category_count}")
        print(f"Database books: {book_count}")
        print(f"Orphaned books: {orphan_count}")

        # The database must contain at least 60 books.
        if book_count < 60:
            raise RuntimeError(
                "Database contains fewer than 60 books."
            )

        # Every book must have a valid category.
        if orphan_count != 0:
            raise RuntimeError(
                "Foreign-key integrity check failed."
            )


# ============================================================
# SQL Queries
# ============================================================

# Store all SQL queries in one dictionary.
# This makes the queries easier to maintain and execute.
SQL_QUERIES = {

    # Query 1:
    # Select books whose GBP price is greater than 40.
    "query_1_select_where": """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE price_gbp > 40
        ORDER BY price_gbp DESC;
    """,

    # Query 2:
    # Sort books by price from highest to lowest
    # and return only the first 10 records.
    "query_2_order_by_limit": """
        SELECT title, price_gbp, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    # Query 3:
    # Return the unique category names.
    "query_3_distinct_categories": """
        SELECT DISTINCT c.category_name
        FROM categories c
        ORDER BY c.category_name;
    """,

    # Query 4:
    # Return books with a rating of either 4 or 5.
    "query_4_in_ratings": """
        SELECT title, rating, price_inr
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, price_inr DESC;
    """,

    # Query 5:
    # Return books with prices between £20 and £40.
    "query_5_between_prices": """
        SELECT title, price_gbp, category_id
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,

    # Query 6:
    # JOIN the books table with the categories table.
    #
    # This demonstrates how the normalized database tables
    # can be combined to produce a useful report.
    "query_6_join_top_rated": """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_gbp,
            b.price_inr,
            b.in_stock
        FROM books b
        INNER JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name ASC,
            b.rating DESC,
            b.price_gbp DESC,
            b.price_inr DESC,
            b.title ASC,
            b.in_stock ASC;
    """,
}


def execute_sql_queries() -> dict[str, pd.DataFrame]:
    """
    Execute all SQL queries and save their results to a text file.

    The results are also returned as pandas DataFrames so that
    they can be reused by other parts of the program.
    """

    # Dictionary used to store the DataFrame result of each query.
    results: dict[str, pd.DataFrame] = {}

    # Open a connection to SQLite.
    with sqlite3.connect(DB_PATH) as connection:

        # Enable foreign-key validation.
        connection.execute("PRAGMA foreign_keys = ON")

        # Open the SQL output file.
        # "w" means a new file is created for every run.
        with SQL_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            # Execute each query stored in SQL_QUERIES.
            for query_name, query in SQL_QUERIES.items():

                # pandas.read_sql() executes the SQL query
                # and directly returns the result as a DataFrame.
                result = pd.read_sql(query, connection)

                # Store the result so it can be accessed later.
                results[query_name] = result

                # Write a clear heading to the output file.
                output_file.write("=" * 80 + "\n")
                output_file.write(f"{query_name}\n")
                output_file.write("=" * 80 + "\n")

                # Save the SQL statement itself.
                output_file.write(
                    "SQL:\n"
                    f"{query.strip()}\n\n"
                )

                # Save the query result.
                output_file.write(
                    "OUTPUT:\n"
                )

                output_file.write(
                    result.to_string(index=False)
                )

                output_file.write("\n\n")

                # Also display the query result in the terminal.
                print(f"\n--- {query_name} ---")
                print(query.strip())
                print(result.to_string(index=False))

    return results


# ============================================================
# pandas read_sql() and pd.merge() Verification
# ============================================================

def pandas_verification() -> None:
    """
    Verify the SQL JOIN using pandas.

    This function demonstrates:
        1. Reading SQL results using pd.read_sql().
        2. Combining tables using pd.merge().
        3. Comparing the SQL JOIN and pandas merge results.
    """

    # Open a database connection.
    with sqlite3.connect(DB_PATH) as connection:

        # ----------------------------------------------------
        # Requirement: demonstrate at least two pd.read_sql()
        # ----------------------------------------------------

        sql_top10 = SQL_QUERIES["query_2_order_by_limit"]
        sql_categories = SQL_QUERIES["query_3_distinct_categories"]

        # Read the top-10 SQL query result into pandas.
        top10_df = pd.read_sql(
            sql_top10,
            connection,
        )

        # Read the category SQL query result into pandas.
        categories_result_df = pd.read_sql(
            sql_categories,
            connection,
        )

        # Save these two examples to an output file.
        with PANDAS_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            output_file.write(
                "RESULT 1 - pd.read_sql(query_2_order_by_limit)\n"
            )

            output_file.write("=" * 80 + "\n")

            output_file.write(
                top10_df.to_string(index=False)
            )

            output_file.write(
                "\n\nRESULT 2 - "
                "pd.read_sql(query_3_distinct_categories)\n"
            )

            output_file.write("=" * 80 + "\n")

            output_file.write(
                categories_result_df.to_string(index=False)
            )

        # ----------------------------------------------------
        # Read the source database tables into pandas
        # ----------------------------------------------------

        # Read all required columns from the books table.
        books_df = pd.read_sql(
            """
            SELECT
                book_id,
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            FROM books
            """,
            connection,
        )

        # Read the category ID and category name.
        categories_df = pd.read_sql(
            """
            SELECT
                category_id,
                category_name
            FROM categories
            """,
            connection,
        )

    # --------------------------------------------------------
    # Reproduce the SQL JOIN using pandas
    # --------------------------------------------------------

    # Match books to categories using category_id.
    #
    # This is the pandas equivalent of:
    #
    # INNER JOIN categories
    #     ON books.category_id = categories.category_id
    merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner",
        validate="many_to_one",
    )

    # Keep the same columns returned by the SQL JOIN.
    merge_result = merge_result[
        [
            "category_name",
            "title",
            "rating",
            "price_gbp",
            "price_inr",
            "in_stock",
        ]
    ].copy()

    # SQLite returns Boolean values as 0 and 1.
    # Convert them back to True/False for comparison.
    merge_result["in_stock"] = (
        merge_result["in_stock"]
        .astype(bool)
    )

    # --------------------------------------------------------
    # Run the same JOIN through SQL
    # --------------------------------------------------------

    with sqlite3.connect(DB_PATH) as connection:

        # Execute Query 6 and store the result as a DataFrame.
        sql_join_result = pd.read_sql(
            SQL_QUERIES["query_6_join_top_rated"],
            connection,
        )

    # Convert SQLite's 0/1 values back into Boolean values.
    sql_join_result["in_stock"] = (
        sql_join_result["in_stock"]
        .astype(bool)
    )

    # --------------------------------------------------------
    # Normalize numeric data types
    # --------------------------------------------------------

    # Both DataFrames should use the same data types before
    # they are compared.
    for df in [sql_join_result, merge_result]:

        # Rating should be an integer.
        df["rating"] = df["rating"].astype("int64")

        # Price should be floating point with two decimal places.
        df["price_gbp"] = (
            df["price_gbp"]
            .astype("float64")
            .round(2)
        )

        df["price_inr"] = (
            df["price_inr"]
            .astype("float64")
            .round(2)
        )

        # Make sure availability is Boolean.
        df["in_stock"] = (
            df["in_stock"]
            .astype(bool)
        )

    # --------------------------------------------------------
    # Sort BOTH results in exactly the same way
    # --------------------------------------------------------

    # The same sorting rules must be applied to both DataFrames
    # before comparing them.
    #
    # Without this step, two identical datasets could appear
    # different simply because their rows are in a different order.
    sort_columns = [
        "category_name",
        "rating",
        "price_gbp",
        "price_inr",
        "title",
        "in_stock",
    ]

    # True means ascending order.
    # False means descending order.
    sort_ascending = [
        True,
        False,
        False,
        False,
        True,
        True,
    ]

    sql_join_result = (
        sql_join_result
        .sort_values(
            by=sort_columns,
            ascending=sort_ascending,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    merge_result = (
        merge_result
        .sort_values(
            by=sort_columns,
            ascending=sort_ascending,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Compare the actual values
    # --------------------------------------------------------

    # Import pandas' testing function.
    # It provides a detailed comparison of two DataFrames.
    from pandas.testing import assert_frame_equal

    try:

        # Check whether both DataFrames contain the same data.
        #
        # check_dtype=False:
        #     Ignore small differences in pandas data types.
        #
        # check_exact=False:
        #     Allow floating-point values to be compared
        #     using tolerance.
        assert_frame_equal(
            sql_join_result,
            merge_result,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )

        # If no exception was raised, the results are equivalent.
        equivalent = True

    except AssertionError as error:

        # If the DataFrames differ, record the failure.
        equivalent = False

        print("\nJOIN comparison failed:")
        print(error)

    # --------------------------------------------------------
    # Create side-by-side comparison output
    # --------------------------------------------------------

    comparison_columns = [
        "category_name",
        "title",
        "rating",
        "price_gbp",
        "price_inr",
        "in_stock",
    ]

    # Place the SQL and pandas results side by side.
    # Prefixes make it clear which result came from which method.
    comparison = pd.concat(
        [
            sql_join_result[
                comparison_columns
            ].add_prefix("pd_read_sql_"),

            merge_result[
                comparison_columns
            ].add_prefix("pd_merge_"),
        ],
        axis=1,
    )

    # Save the complete comparison to a text file.
    with MERGE_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
            "JOIN RESULT FROM pd.read_sql()\n"
        )

        output_file.write("=" * 80 + "\n")

        output_file.write(
            sql_join_result.to_string(index=False)
        )

        output_file.write(
            "\n\nJOIN RESULT FROM pd.merge()\n"
        )

        output_file.write("=" * 80 + "\n")

        output_file.write(
            merge_result.to_string(index=False)
        )

        output_file.write(
            "\n\nSIDE-BY-SIDE COMPARISON\n"
        )

        output_file.write("=" * 80 + "\n")

        output_file.write(
            comparison.to_string(index=False)
        )

        # Record the final comparison result.
        output_file.write(
            "\n\nEquivalent: "
            f"{equivalent}\n"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print(
        "\nJOIN verification using "
        "pd.read_sql vs pd.merge:"
    )

    print(
        f"Equivalent: {equivalent}"
    )

    # If the results do not match, fail the program.
    # This prevents an incorrect JOIN comparison from
    # being treated as a successful submission.
    if not equivalent:
        raise AssertionError(
            "The SQL JOIN and pandas merge results "
            "do not match."
        )


# ============================================================
# Main Program
# ============================================================

def main() -> None:
    """
    Run the complete data pipeline in the required order.

    Pipeline:
        1. Scrape data
        2. Clean and transform data
        3. Save cleaned CSV
        4. Create and populate SQLite database
        5. Execute SQL queries
        6. Verify SQL JOIN using pandas
    """

    print("=" * 80)
    print("MODULE 1 - DATA PIPELINE")
    print("=" * 80)

    # Create the required folders before generating any files.
    ensure_directories()

    # --------------------------------------------------------
    # Step 1: Scrape
    # --------------------------------------------------------

    print("\n[1/6] Scraping...")

    # Scrape all configured categories and create a DataFrame.
    raw_df = scrape_all_categories()

    print(f"Raw rows scraped: {len(raw_df)}")

    print(
        "Raw categories:",
        raw_df["category"].nunique(),
    )

    # --------------------------------------------------------
    # Step 2: Clean and convert
    # --------------------------------------------------------

    print("\n[2/6] Cleaning and converting...")

    # Clean the scraped data and create the required
    # numeric and Boolean fields.
    cleaned_df = clean_data(raw_df)

    print(f"Cleaned rows: {len(cleaned_df)}")

    print(
        "Cleaned categories:",
        cleaned_df["category"].nunique(),
    )

    # Display the final pandas data types.
    print("\nColumn types:")
    print(cleaned_df.dtypes)

    # --------------------------------------------------------
    # Step 3: Save cleaned CSV
    # --------------------------------------------------------

    print("\n[3/6] Saving cleaned CSV...")

    # Save the cleaned DataFrame as a CSV file.
    #
    # index=False prevents pandas from adding its DataFrame
    # index as an extra column.
    cleaned_df.to_csv(
        CSV_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Step 4: Create and populate database
    # --------------------------------------------------------

    print("\n[4/6] Creating and loading SQLite database...")

    # Create a fresh database with the required tables.
    create_database()

    # Insert the cleaned data into the database.
    load_database(cleaned_df)

    # --------------------------------------------------------
    # Step 5: Execute SQL queries
    # --------------------------------------------------------

    print("\n[5/6] Running SQL queries...")

    # Run all required SQL queries and save their outputs.
    execute_sql_queries()

    # --------------------------------------------------------
    # Step 6: Verify using pandas
    # --------------------------------------------------------

    print("\n[6/6] Running pandas verification...")

    # Compare the SQL JOIN result with the equivalent
    # pandas merge result.
    pandas_verification()

    # --------------------------------------------------------
    # Pipeline completed successfully
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    # Display the location of each generated output file.
    print(f"Cleaned data: {CSV_PATH}")
    print(f"SQLite database: {DB_PATH}")
    print(f"SQL outputs: {SQL_OUTPUT_PATH}")
    print(f"pandas outputs: {PANDAS_OUTPUT_PATH}")
    print(f"merge comparison: {MERGE_OUTPUT_PATH}")


# This condition makes sure main() runs only when this file
# is executed directly, and not when it is imported by another file.
if __name__ == "__main__":
    main()