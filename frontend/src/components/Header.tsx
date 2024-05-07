import React, { useEffect, useState } from 'react'
import Link from 'next/link';

import { useTranslation } from 'next-i18next'
import { useTheme } from "next-themes";

import { MdLightMode, MdModeNight, MdLanguage } from "react-icons/md"

import tw from "tailwind-styled-components";

//type

//style
// interface TitleProps {
//     $large: boolean;
// }

// const Title = tw.h1<TitleProps>`
//     ${(p) => (p.$large ? "text-2xl" : "text-base")}
//     text-teal-500
//     font-bold
// `

// const SpecialBlueContainer = styled.section`
//     background-color: #fff;
// `

//react
const Button = tw.button<any>`
    p-2
    font-bold
    text-black
    dark:text-white
    bg-teal-600
    dark:bg-teal-800
    rounded
    cursor-pointer
    shadow
`;

const ButtonIcon = "h-6 w-6 text-gray-100 dark:text-gray-200";

//////////////////////////////////


const ColorMode: React.FC = () => {

    const { systemTheme, theme, setTheme } = useTheme();
    const [mode, setMode] = useState<Mode>(
        {
            check: false,
            name: "light"
        }
    );
    
    const onClickButton = (check: boolean) => {
        const name = check ? "dark" : "light";
        const data: Mode = {
            check: !check,
            name: name
        }
        setMode(data);
        setTheme(name);
    }

    return (
        <Button
            onClick={() => onClickButton(mode.check)}
        >
            {mode.check ? <MdLightMode className={ButtonIcon} /> : <MdModeNight className={ButtonIcon}/>}
        </Button>
    )
}

//////////////////////////////////
const DropDown = tw.div<any>`
    absolute
    mt-12
    right-4
    w-20
    bg-gray-100
    rounded
    divide-y
    divide-gray-100
    shadow
    dark:bg-gray-700
`;

const DropDownUl = tw.div<any>`
    py-1
    text-sm
    text-gray-700
    dark:text-gray-200
    text-center
`;

const DropDownli = tw.div<any>`
    p-2
    hover:bg-gray-400
    cursor-pointer
`;



const LocaleMode: React.FC = () => {
    const selectList = ["ko", "en"];

    const [isOpen, setIsOpen] = useState(false);

    const toggleDropdown = () => {
      setIsOpen(!isOpen);
    };

    return (
        <>
            <Button
                onClick={toggleDropdown}
            >
                <MdLanguage className={ButtonIcon}/>
            </Button>
            {isOpen && (
                <DropDown>
                    <DropDownUl>
                        {
                            selectList.map((data)=>(
                                <DropDownli key={data}>
                                    <Link href={`/${data}`} locale={data}>{data}</Link>
                                </DropDownli>
                            ))
                        }
                    </DropDownUl>
                </DropDown>

            )}
        </>
    )
}


//////////////////////////////////
const Title = tw.h1<any>`
    text-3xl
    text-gray-100
    font-bold
`

const Layout = tw.header<any>`
    flex
    items-center
    justify-between
    w-full
    border-b
    dark:border-gray-600
    bg-teal-800
    dark:bg-teal-600
    py-4
    px-6
`

const SideOptionLayout = tw.div<any>`
    flex
    space-x-2
`

interface Mode {
    check: boolean;
    name: string;
}
const Header: React.FC = () => {
    const { t } = useTranslation('common');

    return (
        <Layout>
            <Title>{t('header.title')}</Title>
            <SideOptionLayout>
                <ColorMode />
                <LocaleMode />
            </SideOptionLayout>
        </Layout>
    )
};

export default Header;