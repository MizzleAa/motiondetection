import type { AppProps } from 'next/app'
import Header from '../components/Header';
import Heads from '../components/Heads';
import Main from '../components/Main';
import Footer from '../components/Footer';

import { ThemeProvider } from "next-themes";
import { appWithTranslation } from 'next-i18next';
import { GetStaticProps } from 'next';
import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
import tw from 'tailwind-styled-components';

//react
const Layout = tw.div<any>`
  flex
  flex-col
  min-h-screen
`;


function App({ Component, pageProps }: AppProps) {
  return (
    <Layout>
      <Heads />
      <ThemeProvider enableSystem={true} attribute="class">
          <Header />
          <Main>
            <Component {...pageProps} />
          </Main>
          <Footer />
      </ThemeProvider>
    </Layout>
  )
}



export default appWithTranslation(App);