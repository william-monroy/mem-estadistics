from __future__ import annotations

from aws_cdk import CfnOutput, IgnoreMode, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_assets as s3_assets
from constructs import Construct

from .config import DeployConfig
from .user_data import build_user_data_script


SUITE_ASSET_EXCLUDES = [
    ".venv",
    ".venv/**",
    ".cache",
    ".cache/**",
    "workspace",
    "workspace/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".DS_Store",
]

DATA_ASSET_EXCLUDES = [
    ".DS_Store",
    "**/.DS_Store",
]


class UltraRunnerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, deploy_config: DeployConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        results_prefix = deploy_config.results_prefix

        suite_asset = s3_assets.Asset(
            self,
            "UltraSuiteAsset",
            path=str(deploy_config.local_suite_dir),
            exclude=SUITE_ASSET_EXCLUDES,
            ignore_mode=IgnoreMode.GLOB,
        )

        data_asset = None
        if deploy_config.include_data_asset:
            data_asset = s3_assets.Asset(
                self,
                "UltraDataAsset",
                path=str(deploy_config.local_data_dir),
                exclude=DATA_ASSET_EXCLUDES,
                ignore_mode=IgnoreMode.GLOB,
            )

        bucket_kwargs = {
            "auto_delete_objects": True,
            "block_public_access": s3.BlockPublicAccess.BLOCK_ALL,
            "encryption": s3.BucketEncryption.S3_MANAGED,
            "enforce_ssl": True,
            "removal_policy": RemovalPolicy.DESTROY,
        }
        if deploy_config.results_bucket_name:
            bucket_kwargs["bucket_name"] = deploy_config.results_bucket_name

        results_bucket = s3.Bucket(self, "ResultsBucket", **bucket_kwargs)

        vpc = ec2.Vpc(
            self,
            "UltraRunnerVpc",
            max_azs=1,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )

        role = iam.Role(
            self,
            "UltraRunnerRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Role used by the AWS ultra suite runner EC2 instance.",
        )
        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"))

        suite_asset.grant_read(role)
        if data_asset:
            data_asset.grant_read(role)
        results_bucket.grant_read_write(role)

        security_group = ec2.SecurityGroup(
            self,
            "UltraRunnerSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
            description="No inbound access. Outbound only for S3, package installs and SSM.",
        )

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            build_user_data_script(
                deploy_config,
                suite_asset=suite_asset,
                data_asset=data_asset,
                results_bucket=results_bucket,
                results_prefix=results_prefix,
                region=self.region,
            )
        )

        instance = ec2.Instance(
            self,
            "UltraRunnerInstance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            associate_public_ip_address=True,
            instance_type=ec2.InstanceType(deploy_config.instance_type),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(cpu_type=ec2.AmazonLinuxCpuType.X86_64),
            require_imdsv2=True,
            role=role,
            security_group=security_group,
            user_data=user_data,
            user_data_causes_replacement=True,
            instance_name=deploy_config.stack_name,
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        deploy_config.volume_size_gb,
                        delete_on_termination=True,
                        encrypted=True,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                    ),
                )
            ],
        )

        CfnOutput(self, "StackName", value=self.stack_name)
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "ResultsBucketName", value=results_bucket.bucket_name)
        CfnOutput(self, "ResultsPrefix", value=results_prefix)
        CfnOutput(self, "ResultsS3Uri", value=f"s3://{results_bucket.bucket_name}/{results_prefix}")
        CfnOutput(
            self,
            "SsmStartSessionCommand",
            value=f"aws ssm start-session --target {instance.instance_id}",
        )
