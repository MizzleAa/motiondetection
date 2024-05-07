import React from 'react';

import tw from "tailwind-styled-components";

//interface

//type
type ChildrenComponentProps = {
    children:React.ReactNode
}


//style
const Layout = tw.div<any>`
    sm:flex md:flex lg:flex xl:flex 2xl:flex
    w-full
    h-full
    flex-grow
`;

const Content = tw.div<any>`
    p-4
    w-full
    xs:h-full
`;
//react
const Main = ({children} : ChildrenComponentProps) => {
    return (
        <Layout>
            <Content>
                {children}
            </Content>
        </Layout>
    )
}

export default Main;