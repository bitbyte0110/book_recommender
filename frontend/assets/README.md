# Assets Directory

This directory is for storing static assets used by the Streamlit frontend:

- **Images**: Book covers, icons, logos
- **CSS**: Custom stylesheets
- **JavaScript**: Custom scripts (if needed)
- **Icons**: Favicon and other icons

## Usage

To add custom assets:

1. Place your files in this directory
2. Reference them in your Streamlit components using relative paths
3. For images: `st.image("frontend/assets/your_image.png")`

## Example

```python
# In your Streamlit component
st.image("frontend/assets/book_cover.png", caption="Book Cover")
```

## Note

For production deployment, consider using external CDNs for better performance.
