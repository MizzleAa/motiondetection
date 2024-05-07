import React from 'react';

import tw from "tailwind-styled-components";

//interface

//type

//style
const Layout = tw.div<any>`
    w-full
    xs:invisible
    p-4 xs:p-0
    border-t
    bg-gray-100
    text-gray-400
    dark:bg-gray-800
    dark:border-gray-600
    mt-auto
    text-center
`;

const Label = tw.label<any>`

`;
//react
const Footer:React.FC = () => {
    return (
        <Layout>
            <Label> Copyright © 2022 airiss Inc. Policy Edit page on GitHub </Label>
        </Layout>
    )
}

export default Footer;