import gradio as gr

def build_app(tagger):
    def search(text, category):
        results, ms = tagger.search(text, category=None if category=="All" else category)
        gallery = [(tagger.image_file(p), f"{p.title or p.id}  ({score:.2f})") for p, score in results]
        return gallery, f"Search took {ms:.0f} ms"
    def upload(image, title):
        if image is None:
            return {}, "Upload a photo first."
        product, top3, ms = tagger.tag_and_add(image, title)
        return top3, f"Added as product {product.id}, tagged by **{product.encoder}** in {ms:.0f} ms"
    with gr.Blocks(title="Visual Catalog") as app:
        with gr.Tab("Search"):
            query = gr.Textbox(label="Search", placeholder="e.g. black leather handbag")
            category = gr.Dropdown(["All"] + tagger.class_names, value="All", label="Category")
            gallery = gr.Gallery(columns=4, height="auto")
            search_info = gr.Markdown()
            query.submit(search, [query, category], [gallery, search_info])

        with gr.Tab("Upload"):
            image = gr.Image(type="pil", label="Product photo")
            title = gr.Textbox(label="Title (optional)")
            button = gr.Button("Tag and add")
            tags = gr.Label(num_top_classes=3, label="Predicted category")
            upload_info = gr.Markdown()
            button.click(upload, [image, title], [tags, upload_info])

    return app