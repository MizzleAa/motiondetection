import React from 'react'

import { useTranslation } from 'next-i18next'
import { GetStaticProps } from 'next';
import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
import WebcamStream from 'components/WebcamStream';

const Index: React.FC = () => {
    const { t } = useTranslation('index');
    return (
        <div>
            <WebcamStream/>
        </div>
    )
}

export const getStaticProps:GetStaticProps = async ({locale}) => ({
    props: {
        ...(await serverSideTranslations(locale ?? "ko", ['common','index'])),
    },
  })

export default Index;