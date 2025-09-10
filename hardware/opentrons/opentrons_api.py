# TODO list
# documentation
# touch_tip
# lights

import requests
import json
import os
from utils.load_save_functions import load_settings
from utils.logger import Logger


class OpentronsAPI:
    def __init__(self):
        settings = load_settings()
        self.ROBOT_IP_ADDRESS = settings["ROBOT_IP"]
        self.LABWARE_DEFINITIONS_FOLDER = "labware\definitions"
        self.HEADERS = {"opentrons-version": "3"}
        self.PROTOCOL_ID = None
        self.RUN_ID = None
        self.COMMANDS_URL = None
        self.X_OFFSET = 0
        self.Y_OFFSET = 0

    ####### protocol management #######
    def upload_protocol(self, protocol: str):
        """
        Method to upload protocol
        """
        url = f"http://{self.ROBOT_IP_ADDRESS}:31950/protocols"
        files = {"files": (protocol, open(f"{protocol}", "rb"))}

        try:
            response = requests.post(url, headers=self.HEADERS, files=files)
            if response.status_code == 201:
                self.logger.info(
                    f"Protocol uploaded succesfully (ID: {self.PROTOCOL_ID})."
                )
                # print("Response:", response.json())
                self.PROTOCOL_ID = response.json()["data"]["id"]
            elif response.status_code == 200:
                self.logger.info("Protocol already uploaded, using existing protocol.")
                self.PROTOCOL_ID = response.json()["data"]["id"]
            else:
                self.logger.error(f"Failed to upload protocol \n {response.text}")
        finally:
            for key, file_obj in files.items():
                file_obj[1].close()

    def create_run(self):
        """
        Method to create a session. Uses the protocol_ID attribute to find the protocol
        """
        url = f"http://{self.ROBOT_IP_ADDRESS}:31950/runs"
        protocol_id_payload = json.dumps({"data": {"protocolId": self.PROTOCOL_ID}})
        response = requests.post(
            url=url, headers=self.HEADERS, data=protocol_id_payload
        )
        try:
            self.RUN_ID = response.json()["data"]["id"]
            self.COMMANDS_URL = (
                f"http://{self.ROBOT_IP_ADDRESS}:31950/runs/{self.RUN_ID}/commands"
            )
            self.logger.info(f"Run created succesfully (ID: {self.RUN_ID}).")
        except:
            self.logger.error(f"Failed to create run! \n {response.text}")

    ####### load functions #######
    def load_pipette(self, name: str, mount: str):
        """
        Loads pipette for a current run.
        """
        command_dict = {
            "data": {
                "commandType": "loadPipette",
                "params": {"pipetteName": name, "mount": mount},
                "intent": "setup",
            }
        }

        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )
        try:
            pipette_id = response.json()["data"]["result"]["pipetteId"]
        except:
            print(f"failed to load pipette {name}: \n{response.text}")
        return pipette_id

    def load_labware(
        self, labware_name: str, labware_file: str, location: int, custom_labware=False
    ):
        """
        loads labware for a current run.
        """
        if custom_labware == True:
            namespace = "custom_beta"
        else:
            namespace = "opentrons"

        command_dict = {
            "data": {
                "commandType": "loadLabware",
                "params": {
                    "location": {"slotName": str(location)},
                    "loadName": labware_file,
                    "namespace": namespace,
                    "version": 1,
                },
                "intent": "setup",
            }
        }
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )
        try:
            response_result = response.json()["data"]["result"]

            ordering = response_result["definition"]["ordering"]
            flatten_ordering = [
                well for row in ordering for well in row
            ]  # flatten ordering, column wise

            labware_info = {
                "labware_name": labware_name,
                "labware_file": labware_file,
                "location": location,
                "labware_id": response_result["labwareId"],
                "ordering": flatten_ordering,
                "well_diameter": response_result["definition"]["wells"]["A1"][
                    "diameter"
                ],
                "max_volume": response_result["definition"]["wells"]["A1"][
                    "totalLiquidVolume"
                ],
                "depth": response_result["definition"]["wells"]["A1"]["depth"],
            }
        except:
            print(
                f"failed to load labware {labware_file} on slot {location}. \n {response.text}"
            )
        return labware_info

    def add_labware_definition(self, labware_definition: str):
        labware_path = self.LABWARE_DEFINITIONS_FOLDER
        with open(f"{labware_path}\{labware_definition}", "rb") as file:
            command_payload = json.dumps({"data": json.load(file)})
        response = requests.post(
            url=f"http://{self.ROBOT_IP_ADDRESS}:31950/runs/{self.RUN_ID}/labware_definitions",
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    def add_all_labware_definitions(self):
        labware_path = self.LABWARE_DEFINITIONS_FOLDER
        try:
            for labware_definition in os.listdir(labware_path):
                self.add_labware_definition(labware_definition)
            self.logger.info("All custom labware definitions added.")
        except:
            self.logger.error(f"Failed to add labware definitions.")

    ####### executable functions #######
    def home(self):
        url = f"http://{self.ROBOT_IP_ADDRESS}:31950/robot/home"
        command_dict = {"target": "robot"}
        command_payload = json.dumps(command_dict)
        response = requests.post(url=url, headers=self.HEADERS, data=command_payload)
        self.logger.info("Robot homed.")

    def delay(self, seconds: float, minutes=0, message=None, intent="setup", log = False):
        command_dict = {
            "data": {
                "commandType": "waitForDuration",
                "params": {"seconds": seconds, "minutes": minutes, "msg": message},
                "intent": intent,
            }
        }

        command_payload = json.dumps(command_dict)

        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )
        if log:
            self.logger.info(f"Delay of {seconds} seconds & {minutes} minutes.")

    def pick_up_tip(
        self,
        pipette_id: str,
        labware_id: str,
        well: str,
        offset: dict = dict(x=0, y=0, z=0),
        intent="setup",
    ):
        "picks up tip on specified pipette"
        command_dict = {
            "data": {
                "commandType": "pickUpTip",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {
                        "origin": "top",
                        "offset": offset,
                        "waitUntilComplete": True,
                    },
                    "pipetteId": pipette_id,
                },
                "intent": intent,
            }
        }
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    def drop_tip(
        self,
        pipette_id: str,
        labware_id="fixedTrash",
        well="A1",
        offset: dict = dict(x=0, y=0, z=0),
        intent="setup",
    ):
        command_dict = {
            "data": {
                "commandType": "dropTip",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {
                        "origin": "top",
                        "offset": offset,
                        "waitUntilComplete": True,
                    },
                    "pipetteId": pipette_id,
                },
                "intent": intent,
            }
        }
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    def aspirate(
        self,
        pipette_id: str,
        labware_id: str,
        volume: int,
        well: str,
        location: str,
        depth=-5,
        flow_rate=100,
        offset: dict = dict(x=0, y=0, z=0),
        intent="setup",
    ):
        offset_depth = offset.copy()
        offset_depth["z"] = depth + offset["z"]
        command_dict = {
            "data": {
                "commandType": "aspirate",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {
                        "origin": "top",
                        "offset": offset_depth,
                    },
                    "flowRate": flow_rate,
                    "volume": volume,
                    "pipetteId": pipette_id,
                },
                "intent": intent,
            }
        }
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    def dispense(
        self,
        pipette_id: str,
        labware_id: str,
        volume: int,
        well: str,
        location: str,
        flow_rate: float,
        depth=-5,
        offset: dict = dict(x=0, y=0, z=0),
        intent="setup",
    ):
        offset_depth = offset.copy()
        offset_depth["z"] = depth + offset["z"]
        command_dict = {
            "data": {
                "commandType": "dispense",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {
                        "origin": "top",
                        "offset": offset_depth,
                    },
                    "flowRate": flow_rate,
                    "volume": volume,
                    "pipetteId": pipette_id,
                },
                "intent": intent,
            }
        }
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    def blow_out(
        self,
        pipette_id: str,
        labware_id: str,
        well: str,
        depth: float = 0,
        offset: dict = dict(x=0, y=0, z=0),
        flow_rate=50,
        intent="setup",
    ):
        offset_depth = offset.copy()
        offset_depth["z"] = depth + offset["z"]
        command_dict = {
            "data": {
                "commandType": "blowout",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {
                        "origin": "top",
                        "offset": offset_depth,
                    },
                    "flowRate": flow_rate,
                    "pipetteId": pipette_id,
                },
                "intent": intent,
            }
        }
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    def move_to_well(
        self,
        pipette_id: str,
        labware_id: str,
        well: str,
        offset: dict = dict(x=0, y=0, z=0),
        speed=None,
        intent="setup",
    ):
        command_dict = {
            "data": {
                "commandType": "moveToWell",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {"origin": "top", "offset": offset},
                    "pipetteId": pipette_id,
                },
                "intent": intent,
            }
        }
        if speed is not None:
            command_dict["data"]["params"]["speed"] = speed
        command_payload = json.dumps(command_dict)
        response = requests.post(
            url=self.COMMANDS_URL,
            headers=self.HEADERS,
            data=command_payload,
            params={"waitUntilComplete": True},
        )

    ####### utils #######
    def _initialize_logger(self):
        settings = load_settings()
        self.logger = Logger(
            name="protocol",
            file_path=f'experiments/{settings["EXPERIMENT_NAME"]}/meta_data',
        )

    def initialise(self):
        self._initialize_logger()
        self.upload_protocol(protocol="hardware\opentrons\protocol_placeholder.py")
        self.create_run()
        self.add_all_labware_definitions()


if __name__ == "__main__":
    pass
