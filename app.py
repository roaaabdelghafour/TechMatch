import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TechMatch",
    page_icon="💻",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(0, 140, 255, 0.18), transparent 30%),
        radial-gradient(circle at 85% 80%, rgba(0, 200, 255, 0.12), transparent 30%),
        linear-gradient(135deg, #07111f, #0b1d33, #06101d);
    color: white;
    }
    .stApp p,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3 {
    color: white;
}

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        margin-top: 1.5rem !important;
    }

    .product-name {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .product-id {
        color: #777;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .product-price {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .spec-title {
        color: #777;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 3px;
    }

    .spec-value {
        font-size: 15px;
        font-weight: 600;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "laptop_rag_documents.csv"
    )

    return df


df = load_data()


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# CREATE FAISS INDEX
# =========================================================

@st.cache_resource
def create_faiss_index(documents):

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


documents = (
    df["document"]
    .fillna("")
    .tolist()
)

index = create_faiss_index(
    documents
)


# =========================================================
# RETRIEVE PRODUCTS
# =========================================================

def retrieve_products(
    query,
    top_k=50
):

    query_embedding = (
        embedding_model.encode(
            [query]
        )
    )

    distances, indices = index.search(
        np.array(
            query_embedding
        ).astype("float32"),
        top_k
    )

    results = df.iloc[
        indices[0]
    ].copy()

    results["distance"] = distances[0]

    return results


# =========================================================
# EXTRACT REQUIREMENTS
# =========================================================

def extract_requirements(question):

    requirements = {
        "brand": None,
        "gpu": None,
        "ram": None,
        "storage": None,
        "max_price": None
    }

    question_lower = question.lower()


    # -------------------------
    # BRAND
    # -------------------------

    brands = [
        "asus",
        "acer",
        "lenovo",
        "hp",
        "dell",
        "msi",
        "apple",
        "huawei",
        "samsung"
    ]

    for brand in brands:

        if brand in question_lower:

            requirements["brand"] = brand

            break


    # -------------------------
    # GPU
    # -------------------------

    gpu_match = re.search(
        r"rtx\s*\d{3,4}",
        question_lower
    )

    if gpu_match:

        requirements["gpu"] = (
            gpu_match.group(0)
            .upper()
            .replace(" ", "")
        )


    # -------------------------
    # RAM
    # -------------------------

    ram_match = re.search(
        r"(\d+)\s*gb\s*ram",
        question_lower
    )

    if ram_match:

        requirements["ram"] = int(
            ram_match.group(1)
        )


    # -------------------------
    # STORAGE
    # -------------------------

    tb_match = re.search(
        r"(\d+(?:\.\d+)?)\s*tb\s*(?:ssd|nvme|storage)?",
        question_lower
    )

    if tb_match:

        requirements["storage"] = int(
            float(
                tb_match.group(1)
            ) * 1024
        )

    else:

        gb_match = re.search(
            r"(\d+)\s*gb\s*(?:ssd|nvme|storage)",
            question_lower
        )

        if gb_match:

            requirements["storage"] = int(
                gb_match.group(1)
            )


    # -------------------------
    # MAX PRICE
    # -------------------------

    price_match = re.search(
        r"(?:under|below|less than|up to)\s*(\d+(?:\.\d+)?)",
        question_lower
    )

    if price_match:

        requirements["max_price"] = float(
            price_match.group(1)
        )


    return requirements


# =========================================================
# PRODUCT QUESTION CHECK
# =========================================================

def is_product_question(question):

    question_lower = question.lower()

    product_keywords = [

        "laptop",
        "notebook",
        "computer",
        "pc",

        "asus",
        "acer",
        "lenovo",
        "hp",
        "dell",
        "msi",
        "apple",
        "huawei",
        "samsung",

        "gaming",

        "ram",
        "ssd",
        "nvme",
        "storage",

        "gpu",
        "rtx",

        "price",
        "budget",

        "processor",
        "cpu",

        "intel",
        "amd",
        "ryzen"
    ]

    return any(
        keyword in question_lower
        for keyword in product_keywords
    )


# =========================================================
# REMOVE ACCESSORIES
# =========================================================

def filter_laptops(results):

    document_text = (
        results["document"]
        .fillna("")
        .str.lower()
    )

    accessory_keywords = [

        "laptop stand",
        "laptop computer desktop tablet stand",
        "computer stand",
        "tablet stand",

        "laptop backpack",
        "laptop bag",
        "laptop case",
        "laptop sleeve",
        "laptop cover",

        "laptop tray",
        "laptop bed tray",
        "laptop table",
        "laptop desk",
        "laptop riser",

        "lap desk",
        "lapboard",

        "laptop cooler",
        "cooling pad",
        "cooling stand",

        "cleaning kit",
        "cleaning suit",
        "laptop cleaning",
        "screen cleaning",
        "laptop cleaner",
        "cleaner for laptop",

        "lcd cleaner",
        "screen cleaner",
        "cleaning cloth",
        "microfiber cloth",

        "laptop accessory",
        "laptop accessories",
        "computer accessory",
        "computer accessories"
    ]

    accessory_pattern = "|".join(
        accessory_keywords
    )

    filtered = results[
        ~document_text.str.contains(
            accessory_pattern,
            case=False,
            na=False,
            regex=True
        )
    ].copy()

    return filtered


# =========================================================
# EXTRACT PRICE
# =========================================================

def extract_price(text):

    match = re.search(
        r"Price:\s*([\d.]+)",
        text
    )

    if match:

        return float(
            match.group(1)
        )

    return None


# =========================================================
# MATCH PRODUCTS
# =========================================================

def match_products(
    question,
    results
):

    requirements = (
        extract_requirements(
            question
        )
    )

    matches = results.copy()


    # -------------------------
    # BRAND
    # -------------------------

    if requirements["brand"] is not None:

        brand = requirements["brand"]

        matches = matches[
            matches["document"].str.contains(
                f"Brand: {brand}",
                case=False,
                na=False
            )
        ]


    # -------------------------
    # GPU
    # -------------------------

    if requirements["gpu"] is not None:

        gpu_value = (
            requirements["gpu"]
            .replace(" ", "")
        )

        matches = matches[
            matches["document"]
            .str.replace(
                " ",
                "",
                regex=False
            )
            .str.contains(
                gpu_value,
                case=False,
                na=False
            )
        ]


    # -------------------------
    # RAM
    # -------------------------

    if requirements["ram"] is not None:

        matches = matches[
            matches["document"].str.contains(
                f'{requirements["ram"]}GB',
                case=False,
                na=False
            )
        ]


    # -------------------------
    # STORAGE
    # -------------------------

    if requirements["storage"] is not None:

        storage_gb = (
            requirements["storage"]
        )

        if storage_gb < 1024:

            storage_pattern = (
                f"{storage_gb}GB"
            )

        else:

            storage_pattern = (
                f"{storage_gb // 1024}TB"
            )

        matches = matches[
            matches["document"].str.contains(
                storage_pattern,
                case=False,
                na=False
            )
        ]


    # -------------------------
    # PRICE
    # -------------------------

    if requirements["max_price"] is not None:

        prices = matches[
            "document"
        ].str.extract(
            r"Price:\s*([\d.]+)"
        )[0].astype(float)

        matches = matches[
            prices <= requirements["max_price"]
        ].copy()


    # -------------------------
    # SORT BY PRICE
    # -------------------------

    matches["price_value"] = (
        matches["document"].apply(
            extract_price
        )
    )

    matches = matches.sort_values(
        by="price_value",
        ascending=True
    )

    return matches


# =========================================================
# EXTRACT FIELD FROM DOCUMENT
# =========================================================

def extract_field(
    document,
    field
):

    document = document.replace(
        "\\n",
        "\n"
    )


    # -------------------------
    # GET REQUESTED FIELD
    # -------------------------

    for line in document.splitlines():

        if line.startswith(
            field + ":"
        ):

            value = line.split(
                ":",
                1
            )[1].strip()

            if (
                value
                and value.lower()
                != "not specified"
            ):

                return value

            break


    # -------------------------
    # GET PRODUCT NAME
    # -------------------------

    name = ""

    for line in document.splitlines():

        if line.startswith(
            "Name:"
        ):

            name = line.split(
                ":",
                1
            )[1].strip()

            break


    # -------------------------
    # GPU FALLBACK
    # -------------------------

    if field == "Gpu Detail":

        match = re.search(
            r"\b(?:RTX|GTX)\s*\d{3,4}\b",
            name,
            re.IGNORECASE
        )

        if match:

            return (
                match.group(0)
                .upper()
                .replace(" ", "")
            )


        if "Qualcomm Adreno GPU" in name:

            return "Qualcomm Adreno GPU"


        if re.search(
            r"Intel[®™]?\s+Arc[™®]?\s+Graphics",
            name,
            re.IGNORECASE
        ):

            return "Intel Arc Graphics"


        if re.search(
            r"Intel[®™]?\s+UHD\s+Graphics",
            name,
            re.IGNORECASE
        ):

            return "Intel UHD Graphics"


        if re.search(
            r"Intel[®™]?\s+Iris\s+Graphics",
            name,
            re.IGNORECASE
        ):

            return "Intel Iris Graphics"


        if re.search(
            r"AMD\s+Radeon",
            name,
            re.IGNORECASE
        ):

            return "AMD Radeon Graphics"


    # -------------------------
    # RAM FALLBACK
    # -------------------------

    if field == "Ram":

        match = re.search(
            r"(\d+)\s*GB\s*(?:DDR\d(?:-\d+)?|LPDDR\dX?|RAM)?",
            name,
            re.IGNORECASE
        )

        if match:

            return (
                f"{match.group(1)}GB"
            )


    # -------------------------
    # STORAGE FALLBACK
    # -------------------------

    if field == "Storage":

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*TB\s*(?:SSD|NVMe)?",
            name,
            re.IGNORECASE
        )

        if match:

            return (
                f"{match.group(1)}TB SSD"
            )


        match = re.search(
            r"(\d+)\s*GB\s*(?:SSD|NVMe)",
            name,
            re.IGNORECASE
        )

        if match:

            return (
                f"{match.group(1)}GB SSD"
            )


    return "Not specified"


# =========================================================
# HEADER
# =========================================================

st.title("💻 TechMatch")

st.caption(
    "Find the right laptop for your needs and budget."
)


# =========================================================
# SEARCH SECTION
# =========================================================

st.subheader(
    "🔎 Find Your Laptop"
)

question = st.text_input(
    "What kind of laptop are you looking for?",
    placeholder=(
        "Example: ASUS laptop with 16GB RAM and "
        "1TB SSD under 60000"
    )
)

search_button = st.button(
    "🔍 Search"
)


# =========================================================
# SEARCH
# =========================================================

if search_button:

    if not question.strip():

        st.warning(
            "Please enter what you are looking for."
        )

    elif not is_product_question(question):

        st.warning(
            "Please enter a laptop-related question."
        )

    else:

        # -------------------------
        # RETRIEVE
        # -------------------------

        results = retrieve_products(
            question,
            top_k=50
        )


        # -------------------------
        # REMOVE ACCESSORIES
        # -------------------------

        results = filter_laptops(
            results
        )


        # -------------------------
        # MATCH REQUIREMENTS
        # -------------------------

        matches = match_products(
            question,
            results
        )


        # -------------------------
        # RESULTS
        # -------------------------

        if matches.empty:

            st.warning(
                "No laptops match all your requirements."
            )

        else:

            st.subheader(
                "💻 Recommended Laptops"
            )

            st.caption(
                f"Found {len(matches)} matching laptop(s)."
            )


            # -------------------------
            # PRODUCT CARDS
            # -------------------------

            for _, row in matches.head(5).iterrows():

                document = row["document"]

                product_id = row[
                    "product_id"
                ]


                name = extract_field(
                    document,
                    "Name"
                )

                price = extract_field(
                    document,
                    "Price"
                )

                gpu = extract_field(
                    document,
                    "Gpu Detail"
                )

                ram = extract_field(
                    document,
                    "Ram"
                )

                storage = extract_field(
                    document,
                    "Storage"
                )


                image_url = row["image"]

                product_url = row["url"]


                # -------------------------
                # PRODUCT CONTAINER
                # -------------------------

                with st.container(
                    border=True
                ):

                    col1, col2 = st.columns(
                        [1, 2]
                    )


                    # -------------------------
                    # IMAGE
                    # -------------------------

                    with col1:

                        if (
                            pd.notna(image_url)
                            and str(
                                image_url
                            ).strip() != ""
                        ):

                            try:

                                st.image(
                                    str(
                                        image_url
                                    ).strip(),
                                    use_container_width=True
                                )

                            except Exception:

                                st.info(
                                    "Product image unavailable."
                                )


                    # -------------------------
                    # PRODUCT INFO
                    # -------------------------

                    with col2:

                        st.markdown(
                            f'<div class="product-name">{name}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="product-id">Product ID: {product_id}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="product-price">💰 {price}</div>',
                            unsafe_allow_html=True
                        )


                        # -------------------------
                        # SPECIFICATIONS
                        # -------------------------

                        spec1, spec2, spec3 = st.columns(
                            3
                        )


                        with spec1:

                            st.markdown(
                                '<div class="spec-title">RAM</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div class="spec-value">🧠 {ram}</div>',
                                unsafe_allow_html=True
                            )


                        with spec2:

                            st.markdown(
                                '<div class="spec-title">Storage</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div class="spec-value">💾 {storage}</div>',
                                unsafe_allow_html=True
                            )


                        with spec3:

                            st.markdown(
                                '<div class="spec-title">GPU</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div class="spec-value">🎮 {gpu}</div>',
                                unsafe_allow_html=True
                            )


                        # -------------------------
                        # PRODUCT LINK
                        # -------------------------

                        if (
                            pd.notna(product_url)
                            and str(
                                product_url
                            ).strip() != ""
                        ):

                            st.link_button(
                                "🛒 View Product",
                                str(
                                    product_url
                                ).strip()
                            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#888;
        padding-top:40px;
        font-size:13px;
    ">
        TechMatch • AI-Powered Laptop Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)