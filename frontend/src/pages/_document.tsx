
import Document, { Html, Head, Main, NextScript, DocumentContext } from 'next/document'
import { ServerStyleSheet } from "styled-components";

//@ts-ignore
import tailwindCSS from "!raw-loader!../styles/tailwindSSR.css";

class MyDocument extends Document {
  static async getInitialProps(ctx: DocumentContext) {
    const sheet = new ServerStyleSheet();
    const originalRenderPage = ctx.renderPage;

    try {
      ctx.renderPage = () =>
        originalRenderPage({
          enhanceApp: (App) => (props) =>
            sheet.collectStyles(<App {...props} />),
        });

      const initialProps = await Document.getInitialProps(ctx);
      return {
        ...initialProps,
        styles: [
          initialProps.styles,
          <style
            key="custom"
            dangerouslySetInnerHTML={{
              __html: tailwindCSS,
            }}
          />,
          sheet.getStyleElement(),
        ],
      };
    } finally {
      sheet.seal();
    }
  }
}

export default MyDocument;