"""MkDocs hook: keep heading IDs valid as CSS selectors.

Headings such as "1. Introduction" get the auto ID "1-introduction".
IDs that start with a digit are valid HTML but invalid in
`querySelector('#1-introduction')`, which the theme's table-of-contents
scrollspy uses, so the page threw on load. Prefix those IDs with "sec-".
"""
from markdown.extensions.toc import slugify as _default_slugify


def slugify(value, separator):
    slug = _default_slugify(value, separator)
    return f"sec{separator}{slug}" if slug[:1].isdigit() else slug


def on_config(config, **kwargs):
    config.mdx_configs.setdefault("toc", {})["slugify"] = slugify
    return config
