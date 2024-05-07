import { useEffect, useRef, useState } from "react";
import { useTranslation } from 'next-i18next'

import tw from "tailwind-styled-components";
import { FaPlay, FaStop } from "react-icons/fa";

const Button = tw.button<any>`
    p-2
    font-bold
    text-black
    dark:text-white
    bg-gray-300
    dark:bg-gray-600
    rounded
    cursor-pointer

    hover:bg-gray-400
    hover:dark:bg-gray-500
`;

const ButtonIcon = "h-4 w-4 text-gray-700 dark:text-gray-200";

const Layout = tw.div<any>`
    space-y-4
    w-full
`;

const MenuLayout = tw.div<any>`
    bg-gray-100
    dark:bg-gray-700
    rounded
    w-full
`;


const MenuInnerLayout = tw.div<any>`
    flex
    items-center
    justify-between
    w-full
`;

const MenuUl = tw.div<any>`
    m-2 flex space-x-4
    justify-between
    items-center
`;

const MenuLi = tw.div<any>`
    w-full
    flex items-start justify-center
    text-gray-300
    hover:text-white
    space-x-2
`;

const ViewTitle = tw.div<any>`
    bg-gray-100
    dark:bg-gray-700
    p-2
    font-bold
    rounded
    mb-4
`;

const ViewLayout = tw.div<any>`
    flex 
    space-x-4

`;


const ViewInnverLayout = tw.div<any>`
    w-full
    h-full
    rounded
`;

const ViewSplitLayout = tw.div<any>`
    grid 
    grid-cols-1
    gap-4
`;


const ViewImage = tw.img<any>`
    w-full
    h-full
    rounded
`;

const ViewWarnImage = tw.img<any>`
    w-full
    h-full
    rounded
    dark:border-red-400
    border-red-600
`;


const LogTextArea = tw.textarea<any>`
    w-full 
    px-4 py-2 
    rounded 
    bg-gray-100
    dark:bg-gray-700
    border-teal-300
    focus:outline-none focus:border-teal-900
`;

const Hr = tw.hr<any>`
    border
    border-gray-200
    dark:border-gray-600
    h-8
`;

const WarnDiv = tw.div<any>`
    absolute 
    top-0 left-0 
    w-full h-full 
    bg-red-400 
    opacity-30 
    flex items-center justify-center
    z-50
`;

const WarnAnimationDiv = tw.div<any>`
    w-1/2 h-1/2 bg-red-600 animate-ping 
`;

interface WebCamProps {
    index: number,
    isError: boolean,
    imageRef: React.RefObject<HTMLImageElement>,
    canvasRef: React.RefObject<HTMLCanvasElement>
}

const WebCamImage = ({ index, isError, imageRef, canvasRef }: WebCamProps) => {


    return (
        <div className="w-full h-full relative">
            {
                isError ?
                    <div>
                        <WarnDiv>
                            <WarnAnimationDiv />
                        </WarnDiv>
                        {/* <canvas className="z-10 absolute top-0 left-0 w-full h-full rounded" ref={canvasRef}/> */}
                        <ViewWarnImage key={index} ref={imageRef} src="/images/default.jpg" alt={`Image ${index + 1}`} />
                    </div>
                    :
                    <div>
                        <ViewImage key={index} ref={imageRef} src="/images/default.jpg" alt={`Image ${index + 1}`} />
                    </div>
            }
        </div>
    );
}


interface WebCamIndexProps {
    cameraId: number,
    port:number
}

const Webcam = ( {cameraId, port}:WebCamIndexProps) => {
    const { t } = useTranslation('common');

    const originImageRef = useRef<HTMLImageElement>(null);
    const skeletronImageRef = useRef<HTMLImageElement>(null);
    // const skeletronImageRefs = [
    //     useRef<HTMLImageElement>(null),
    //     useRef<HTMLImageElement>(null),
    // ];
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [isAction, setIsAction] = useState<boolean>(false);
    const [isSkeletronAction, setIsSkeletronAction] = useState<boolean>(true);
    const [isSelected, setIsSelected] = useState<number>(0);

    const [isError, setIsError] = useState<boolean>(false);
    const [log, setLog] = useState<string>('');
    const [fps, setFps] = useState<string>('');


    useEffect(() => {
        if (isAction) {
            let socket = new WebSocket(`ws://127.0.0.1:${port}/video/${cameraId}`);

            socket = ConnectCamera(socket);

            return () => {
                socket.close();
            };
        }
    }, [isAction]);

    const ConnectSampleCamera = () => {
        const socket = new WebSocket("ws://192.168.0.11:8000/video/0");
 
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            // console.log(typeof(data["skeletron_image"]));
            // originImageRef.current!.src = "data:image/jpeg;base64," + data["origin_image"];
            // skeletronImageRef.current!.src = "data:image/jpeg;base64," + data["skeletron_image"];
            setLog(data["information"]);
            // setFps(data["fps"]);
            // setProcessTime(data["process_time"]);
            // GridRandomPointCanvas();
        };

        return () => {
            socket.close();
        };
    }

    const ConnectCamera = (socket : WebSocket) => {
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            skeletronImageRef.current!.src =  "data:image/jpeg;base64," + data["skeletron_image"];
            setLog(data["information"]);
            setFps(data["fps"])
            setIsError(data["error"])
        };

        return socket;
    }

    const GridRandomPointCanvas = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const context = canvas.getContext('2d');
        if (!context) return;
        context.clearRect(0, 0, canvas.width, canvas.height);

        const gridSize = getRandomInteger(10,20); 
        const gridColor = '#ccc'; 

        context.beginPath();
        for (let y = 0; y <= canvas.height; y += gridSize) {
            context.moveTo(0, y);
            context.lineTo(canvas.width, y);
        }

        context.strokeStyle = gridColor;
        context.stroke();
        context.closePath();
    }

    const getRandomInteger = (min:number, max:number) => {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    const onClickIsAction = () => {
        setIsAction(!isAction);
    }

    const onClickImages = (index: number) => {
        setIsSelected(index);
    }

    const onHandleChangeLog = (event: any) => {
        // setLog(event.target.value);
    };

    return (
        <Layout>
            <MenuLayout>
                <MenuInnerLayout>
                    <MenuUl>
                        <MenuLi>
                            <Button onClick={onClickIsAction}>
                                {isAction ?
                                    <FaStop className={ButtonIcon} /> :
                                    <FaPlay className={ButtonIcon} />
                                }
                            </Button>
                            <Hr />
                        </MenuLi>
                        <MenuLi>
                            <label>FPS</label>
                            <label>{fps}</label>
                        </MenuLi>
                    </MenuUl>
                </MenuInnerLayout>
            </MenuLayout>

            <ViewLayout>
                {/* <ViewInnverLayout>
                    <ViewTitle>{t('webcamstream.origin')}</ViewTitle>
                    <ViewImage ref={originImageRef} src="/images/default.jpg" alt="" />
                </ViewInnverLayout> */}
                <ViewInnverLayout>
                    <ViewTitle>{t('webcamstream.detection')} - {cameraId}</ViewTitle>
                    <ViewSplitLayout>
                        <WebCamImage key={cameraId} index={cameraId} isError={isError} imageRef={skeletronImageRef} canvasRef={canvasRef}/>
                    </ViewSplitLayout>
                </ViewInnverLayout>
            </ViewLayout>

            <MenuLayout>
                <MenuInnerLayout>
                    <LogTextArea
                        value={log}
                        onChange={onHandleChangeLog}>
                    </LogTextArea>
                </MenuInnerLayout>
            </MenuLayout>
        </Layout>
    );
};


const WebcamStreamLayout = tw.div<any>`
    grid 
    md:grid-cols-2
    sm:grid-cols-1
    gap-4
`;


const WebcamStream = () => {
    return (
        <WebcamStreamLayout>
            <Webcam cameraId={0} port={8000}/>
            <Webcam cameraId={1} port={8000}/>

        </WebcamStreamLayout>
    );
}

export default WebcamStream;