# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
CSS and HTML content for the MapAnything Gradio application.
This module contains all the CSS styles and HTML content blocks
used in the Gradio interface.
"""

# CSS Styles for the Gradio interface
GRADIO_CSS = """
.custom-log * {
    font-style: italic;
    font-size: 22px !important;
    background-image: linear-gradient(120deg, #ffb366 0%, #ffa366 60%, #ff9966 100%);
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: bold !important;
    color: transparent !important;
    text-align: center !important;
}

.example-log * {
    font-style: italic;
    font-size: 16px !important;
    background-image: linear-gradient(120deg, #ffb366 0%, #ffa366 60%, #ff9966 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}

#my_radio .wrap {
    display: flex;
    flex-wrap: nowrap;
    justify-content: center;
    align-items: center;
}

#my_radio .wrap label {
    display: flex;
    width: 50%;
    justify-content: center;
    align-items: center;
    margin: 0;
    padding: 10px 0;
    box-sizing: border-box;
}

/* Align navigation buttons with dropdown bottom */
.navigation-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
}

.navigation-row > div:nth-child(1),
.navigation-row > div:nth-child(3) {
    align-self: flex-end !important;
}

.navigation-row > div:nth-child(2) {
    flex: 1 !important;
}

/* Make thumbnails clickable with pointer cursor */
.clickable-thumbnail img {
    cursor: pointer !important;
}

.clickable-thumbnail:hover img {
    cursor: pointer !important;
    opacity: 0.8;
    transition: opacity 0.3s ease;
}

/* Make thumbnail containers narrower horizontally */
.clickable-thumbnail {
    padding: 5px 2px !important;
    margin: 0 2px !important;
}

.clickable-thumbnail .image-container {
    margin: 0 !important;
    padding: 0 !important;
}

.scene-info {
    text-align: center !important;
    padding: 5px 2px !important;
    margin: 0 !important;
}
"""


def get_header_html(logo_base64=None):
    """
    Generate the main header HTML with logo and title.

    Args:
        logo_base64 (str, optional): Base64 encoded logo image

    Returns:
        str: HTML string for the header
    """
    logo_style = "display: none;" if not logo_base64 else ""
    logo_src = logo_base64 or ""

    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="{logo_src}" alt="WAI Logo" style="height: 40px; margin-right: 15px; {logo_style}">
        <h1 style="margin: 0;"><span style="color: #ffb366;">MapAnything:</span> <span style="color: #555555;">Metric 3D Scene Reconstruction</span></h1>
    </div>
    <p>
    <a href="https://github.com/facebookresearch/map-anything">🌟 GitHub Repository</a> |
    <a href="https://map-anything.github.io/">🚀 Project Page</a>
    </p>
    """


def get_description_html():
    """
    Generate the main description and getting started HTML.

    Returns:
        str: HTML string for the description
    """
    return """
    """


def get_acknowledgements_html():
    """
    Generate the acknowledgements section HTML.

    Returns:
        str: HTML string for the acknowledgements
    """
    return """

    """


def get_gradio_theme():
    """
    Get the configured Gradio theme.

    Returns:
        gr.themes.Base: Configured Gradio theme
    """
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c100="#ffedd5",
            c200="#ffddb3",
            c300="rgba(242.78125, 182.89427563548466, 120.32579495614034, 1)",
            c400="#fb923c",
            c50="#fff7ed",
            c500="#f97316",
            c600="#ea580c",
            c700="#c2410c",
            c800="#9a3412",
            c900="#7c2d12",
            c950="#6c2e12",
        ),
        secondary_hue="amber",
    )


# Measure tab instructions HTML
MEASURE_INSTRUCTIONS_HTML = """
### Click on the image to measure the distance between two points.
"""
