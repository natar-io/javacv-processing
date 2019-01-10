# Maintainer: RealityTech <laviole@rea.lity.tech>
pkgname=javacv-processing-git
pkgver=0.1
pkgrel=1
epoch=
pkgdesc=""
arch=('x86_64')
url=""
license=('GPL')
groups=()
depends=('java-runtime')
makedepends=('maven' 'jdk8-openjdk' 'git')
#provides=("${pkgname%-git}")
#conflicts=("${pkgname%-git}")
replaces=()
#backup=()
#options=()
#install=
changelog=
# source=($pkgname-$pkgver.tar.gz
#         $pkgname-$pkgver.patch)
# noextract=()
# md5sums=()

prepare() {
  find ${startdir} -path ${startdir}/pkg -prune -o -name pom.xml -exec perl -pi -e "s/(?<=\<\!-- PKGVER -->)(\d+(\.\d+)+)(?=\<\!-- \/PKGVER -->)/${pkgver}/g" {} \;
}

build() {
  #  cd "$srcdir/$pkgname-$pkgver"
  cd ${startdir}
  mvn package install ${MVN_OPTS}
}

package() {
  local name='javacv-processing'

  install -m644 -D ${startdir}/target/${name}-${pkgver}-jar-with-dependencies.jar ${pkgdir}/usr/share/java/natar/${name}/${pkgver}/${name}-${pkgver}.jar
  cd ${pkgdir}/usr/share/java/natar
  ln -s ${name}/${pkgver}/${name}-${pkgver}.jar $name.jar
}

# installOne() {
#     local name=$1
#     install -m644 -D ${startdir}/target/${name}-${pkgver}.jar ${pkgdir}/usr/share/java/natar/${name}/${pkgver}/${name}-${pkgver}.jar
#     cd ${pkgdir}/usr/share/java/natar
#     ln -s ${name}/${pkgver}/${name}-${pkgver}.jar $name.jar
# }

## inspiration: https://github.com/diet4j/diet4j-examples/blob/master/PKGBUILD
